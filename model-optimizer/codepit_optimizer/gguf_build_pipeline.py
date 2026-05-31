"""Real GGUF build pipeline: HF model -> GGUF convert -> quantize.

Replaces the JSON `gguf_fixture.v1` stub with a genuine `.gguf` binary produced
by llama.cpp (`convert_hf_to_gguf.py` then `llama-quantize`). The subprocess is
injected as a `Runner` seam so command assembly, quantization-profile mapping,
GGUF-magic validation, checksum, and provenance are unit-testable without the
real binaries; the deploy path supplies a real `subprocess.run` + a pinned
llama.cpp toolchain.

Fails closed (raises `GgufBuildError`) whenever a tool errors or the produced
file is not a real GGUF, so a failed build can never yield a verifier
submission. Exact convert/quantize flags are pinned to the fleet's llama.cpp
build; validate against that version at integration.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, run
from typing import Any

GGUF_MAGIC = b"GGUF"

Runner = Callable[..., CompletedProcess]

# Produces a real GGUF at `out_gguf` for the given base model + profile and
# returns a provenance dict. Injected into the packager so the build path is a
# swappable seam (fixture when absent, real llama.cpp when present).
GgufBuilder = Callable[..., dict[str, Any]]


class GgufBuildError(RuntimeError):
    """Raised when GGUF conversion, quantization, or validation cannot complete."""


# Accepted quantization profiles -> the llama-quantize type token. CPU-feasible
# targets for the tiny-chat (ollama-gguf-local) lane.
_QUANT_TYPE_BY_PROFILE: dict[str, str] = {
    "q8_0": "Q8_0",
    "q6_k": "Q6_K",
    "q5_k_m": "Q5_K_M",
    "q4_k_m": "Q4_K_M",
    "f16": "F16",
}

#: Profiles a tiny-chat agent may actually submit on the hosted-base path.
#: Deliberately excludes ``f16`` (quantizing an FP16 base to F16 is a
#: byte-identical no-op that would resubmit the base as fake "work"). Mirrors
#: ``orchestrator.TINY_CHAT_ALLOWED_QUANTIZATION_PROFILES`` — kept here too to
#: avoid a circular import (orchestrator imports this module).
TINY_CHAT_HOSTED_BASE_PROFILES: tuple[str, ...] = ("q4_k_m", "q5_k_m", "q8_0")


@dataclass(frozen=True)
class GgufToolchain:
    convert_script: str
    quantize_bin: str
    python_bin: str = "python3"


@dataclass(frozen=True)
class GgufBuildResult:
    gguf_path: Path
    quantization_profile: str
    quant_type: str
    sha256_hex: str
    size_bytes: int
    provenance: dict[str, Any]


def resolve_quant_type(quantization_profile: str) -> str:
    quant_type = _QUANT_TYPE_BY_PROFILE.get(quantization_profile.strip().lower())
    if quant_type is None:
        valid = ", ".join(sorted(_QUANT_TYPE_BY_PROFILE))
        raise GgufBuildError(
            f"unsupported quantization profile: {quantization_profile!r}. "
            f"Valid profiles: {valid}",
        )
    return quant_type


def build_convert_command(
    toolchain: GgufToolchain,
    model_dir: Path,
    out_gguf: Path,
    *,
    out_type: str = "f16",
) -> list[str]:
    return [
        toolchain.python_bin,
        toolchain.convert_script,
        str(model_dir),
        "--outfile",
        str(out_gguf),
        "--outtype",
        out_type,
    ]


def build_quantize_command(
    toolchain: GgufToolchain,
    src_gguf: Path,
    out_gguf: Path,
    quant_type: str,
) -> list[str]:
    return [toolchain.quantize_bin, str(src_gguf), str(out_gguf), quant_type]


def validate_gguf_magic(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            header = handle.read(len(GGUF_MAGIC))
    except OSError as error:
        raise GgufBuildError(f"cannot read produced GGUF {path}: {error}") from error
    if header != GGUF_MAGIC:
        raise GgufBuildError(
            f"produced file {path} is not a GGUF binary (bad magic: {header!r})",
        )


def build_gguf(
    *,
    toolchain: GgufToolchain,
    base_model_ref: str,
    base_model_dir: Path,
    quantization_profile: str,
    work_dir: Path,
    runner: Runner = run,
) -> GgufBuildResult:
    quant_type = resolve_quant_type(quantization_profile)
    work_dir.mkdir(parents=True, exist_ok=True)

    f16_gguf = work_dir / "model-f16.gguf"
    quantized_gguf = work_dir / f"model-{_safe_name(quantization_profile)}.gguf"

    _run_tool(
        runner,
        build_convert_command(toolchain, base_model_dir, f16_gguf, out_type="f16"),
    )
    validate_gguf_magic(f16_gguf)

    _run_tool(
        runner,
        build_quantize_command(toolchain, f16_gguf, quantized_gguf, quant_type),
    )
    validate_gguf_magic(quantized_gguf)

    f16_sha256 = _sha256_file(f16_gguf)
    gguf_sha256 = _sha256_file(quantized_gguf)
    provenance = {
        "schema": "codepit.tiny_chat.gguf_build.v1",
        "base_model_ref": base_model_ref,
        "quantization_profile": quantization_profile,
        "quant_type": quant_type,
        "converter": toolchain.convert_script,
        "quantizer": toolchain.quantize_bin,
        "intermediate_f16_sha256": f16_sha256,
        "gguf_sha256": gguf_sha256,
    }

    return GgufBuildResult(
        gguf_path=quantized_gguf,
        quantization_profile=quantization_profile,
        quant_type=quant_type,
        sha256_hex=gguf_sha256,
        size_bytes=quantized_gguf.stat().st_size,
        provenance=provenance,
    )


#: Env var pointing at an explicit ``llama-quantize`` binary. When unset, the
#: hosted-base builder auto-detects it on PATH (one ``brew install llama.cpp``).
QUANTIZE_BIN_ENV = "CODEPIT_GGUF_QUANTIZE_BIN"


def resolve_quantize_bin(
    *,
    env: dict[str, str] | None = None,
    which: Callable[[str], str | None] = shutil.which,
) -> str:
    """Resolve the ``llama-quantize`` binary for the lightweight hosted-base path.

    Prefers an explicit ``CODEPIT_GGUF_QUANTIZE_BIN`` (so operators can pin a
    specific build), else auto-detects ``llama-quantize`` on PATH. Fails closed
    with the one-line fix so a fresh agent is never left guessing — and the
    tiny-chat lane never ships a fixture in lieu of a real quantize.
    """
    resolved_env = env if env is not None else os.environ
    explicit = resolved_env.get(QUANTIZE_BIN_ENV)
    if explicit:
        return explicit
    detected = which("llama-quantize")
    if detected:
        return detected
    raise GgufBuildError(
        "no llama-quantize binary found: set "
        f"{QUANTIZE_BIN_ENV} or install one on PATH (`brew install llama.cpp`).",
    )


#: Resolves the hosted FP16 base GGUF to a local path. Injected as a seam so the
#: download/cache logic (orchestrator) stays separate from the quantize step and
#: both are testable without network.
BaseProvider = Callable[[Path], Path]


def build_hosted_base_quantized_gguf(
    *,
    base_provider: BaseProvider,
    quantize_bin: str,
    quantization_profile: str,
    out_gguf: Path,
    runner: Runner = run,
) -> dict[str, Any]:
    """Quantize-only build for the lightweight path (slice C / #274).

    Resolves the hosted FP16 base via ``base_provider`` (download+cache seam),
    then runs ``llama-quantize`` only — no convert step, no torch, no HF
    download. Restricted to ``TINY_CHAT_HOSTED_BASE_PROFILES`` and rejects
    ``f16`` (a no-op). Fails closed (``GgufBuildError``) if the output is not a
    real GGUF, so a broken quantize can never become a submission. Returns a
    provenance dict recording both the base and produced artifact hashes.
    """
    profile = quantization_profile.strip().lower()
    if profile not in TINY_CHAT_HOSTED_BASE_PROFILES:
        valid = ", ".join(TINY_CHAT_HOSTED_BASE_PROFILES)
        raise GgufBuildError(
            f"quantization profile {quantization_profile!r} is not permitted on the "
            f"tiny-chat hosted-base path. Valid profiles: {valid} "
            "(f16 is rejected: quantizing the FP16 base to F16 is a no-op).",
        )
    quant_type = resolve_quant_type(profile)

    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    work_dir = out_gguf.parent / ".hosted-base"
    work_dir.mkdir(parents=True, exist_ok=True)
    base_path = base_provider(work_dir / "base-f16.gguf")
    validate_gguf_magic(base_path)

    toolchain = GgufToolchain(convert_script="", quantize_bin=quantize_bin)
    _run_tool(runner, build_quantize_command(toolchain, base_path, out_gguf, quant_type))
    validate_gguf_magic(out_gguf)

    return {
        "schema": "codepit.tiny_chat.hosted_base_quantize.v1",
        "method": "quantize-only-from-hosted-base",
        "quantization_profile": profile,
        "quant_type": quant_type,
        "quantizer": quantize_bin,
        "base_sha256": _sha256_file(base_path),
        "gguf_sha256": _sha256_file(out_gguf),
    }


def make_env_gguf_builder() -> GgufBuilder | None:
    """Mode gate: return a real GGUF builder when the llama.cpp toolchain + base
    model dir are configured via env, else ``None`` so the packager keeps the
    deterministic fixture path (CI). The returned builder produces the GGUF in a
    work dir then moves it to the packager's target path.
    """
    convert_script = os.environ.get("CODEPIT_GGUF_CONVERT_SCRIPT")
    quantize_bin = os.environ.get("CODEPIT_GGUF_QUANTIZE_BIN")
    base_model_dir = os.environ.get("CODEPIT_GGUF_BASE_MODEL_DIR")
    if not (convert_script and quantize_bin and base_model_dir):
        return None

    toolchain = GgufToolchain(
        convert_script=convert_script,
        quantize_bin=quantize_bin,
        python_bin=os.environ.get("CODEPIT_GGUF_PYTHON_BIN", "python3"),
    )
    resolved_base_dir = Path(base_model_dir)

    def builder(*, base_model_ref: str, quantization_profile: str, out_gguf: Path) -> dict[str, Any]:
        result = build_gguf(
            toolchain=toolchain,
            base_model_ref=base_model_ref,
            base_model_dir=resolved_base_dir,
            quantization_profile=quantization_profile,
            work_dir=out_gguf.parent / ".gguf-build",
            runner=run,
        )
        shutil.move(str(result.gguf_path), str(out_gguf))
        return result.provenance

    return builder


def _run_tool(runner: Runner, command: list[str]) -> None:
    try:
        runner(command, check=True)
    except GgufBuildError:
        raise
    except Exception as error:  # subprocess error, missing binary, etc.
        raise GgufBuildError(
            f"GGUF build step failed: {command[0]} ({error})",
        ) from error


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip(".-")
    return cleaned or "artifact"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
