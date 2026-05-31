"""Build and publish the FP16 *base* GGUF for the tiny-chat lane (slice A / #271).

This produces the unoptimized FP16 GGUF that CodePit hosts on Cloudflare R2 so a
lightweight external agent only needs `llama-quantize` (one `brew install`) to
quantize it into a submission. We host the *base* (not a pre-quantized artifact)
so the agent's real job remains choosing + applying a quantization profile, and
so the artifact's provenance stays under CodePit control.

The heavyweight legs are injected as seams: the HF->FP16 GGUF convert subprocess
(a `Runner`) and the R2 object store (an `ObjectStore`). This keeps provenance,
GGUF-magic validation, the stable R2 key, and sha256 upload reconciliation
unit-testable without the real toolchain, model download, or network. Fails
closed (raises `BaseGgufBuildError`) on any non-GGUF output so a broken build can
never be published.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from subprocess import run as subprocess_run
from typing import Any, Callable

from .gguf_build_pipeline import (
    GgufBuildError,
    GgufToolchain,
    build_convert_command,
    validate_gguf_magic,
)

#: A convert subprocess seam: ``(command, *, check) -> None``. The deploy path
#: supplies ``subprocess.run`` + a pinned llama.cpp ``convert_hf_to_gguf.py``;
#: tests inject a fake that writes prearranged bytes.
Runner = Callable[..., Any]


#: The stable, documented R2 object key for the tiny-chat FP16 base GGUF. The
#: engine 302 route (#273) resolves this key; the kit's default base URL (#274)
#: points at that route. Keep this constant — agents cache by it.
TINY_CHAT_BASE_GGUF_KEY = "base-models/tiny-chat/qwen2.5-0.5b-instruct/fp16.gguf"


class BaseGgufBuildError(RuntimeError):
    """Raised when the FP16 base GGUF build, validation, or publish cannot complete."""


def default_base_gguf_key() -> str:
    """Return the stable R2 key for the hosted tiny-chat FP16 base GGUF."""
    return TINY_CHAT_BASE_GGUF_KEY


@dataclass(frozen=True)
class BaseGgufBuildResult:
    gguf_path: Path
    sha256_hex: str
    size_bytes: int
    provenance: dict[str, Any]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_base_gguf_provenance(
    artifact: Path,
    *,
    base_model_ref: str,
    source_revision: str,
    llama_cpp_version: str,
) -> dict[str, Any]:
    """Record the auditable identity of a built FP16 base GGUF.

    The recorded sha256 is the trust anchor the kit verifies after downloading
    the hosted base, so it must be computed over the exact bytes on disk.
    """
    return {
        "schema": "codepit.tiny_chat.base_gguf.v1",
        "base_model_ref": base_model_ref,
        "source_revision": source_revision,
        "llama_cpp_version": llama_cpp_version,
        "artifact_kind": "gguf-fp16",
        "sha256_hex": _sha256_file(artifact),
        "size_bytes": artifact.stat().st_size,
    }


def build_base_gguf_fp16(
    *,
    base_model_dir: Path,
    out_gguf: Path,
    base_model_ref: str,
    source_revision: str,
    convert_script: str,
    llama_cpp_version: str,
    python_bin: str = "python3",
    runner: Runner = subprocess_run,
) -> BaseGgufBuildResult:
    """Convert a local HF base model checkout to an FP16 GGUF and record provenance.

    Reuses the canonical convert-command builder and GGUF-magic validator so the
    base artifact is produced exactly like the agent-side pipeline's f16 step.
    Fails closed (``BaseGgufBuildError``) if the convert step errors or the output
    is not a real GGUF, so a broken build can never be published.
    """
    toolchain = GgufToolchain(
        convert_script=convert_script,
        quantize_bin="",  # unused: base build is convert-only (no quantization)
        python_bin=python_bin,
    )
    out_gguf.parent.mkdir(parents=True, exist_ok=True)
    command = build_convert_command(toolchain, base_model_dir, out_gguf, out_type="f16")

    try:
        runner(command, check=True)
    except Exception as error:  # subprocess error, missing binary, etc.
        raise BaseGgufBuildError(
            f"FP16 convert step failed: {command[0]} ({error})",
        ) from error

    try:
        validate_gguf_magic(out_gguf)
    except GgufBuildError as error:
        raise BaseGgufBuildError(str(error)) from error

    provenance = compute_base_gguf_provenance(
        out_gguf,
        base_model_ref=base_model_ref,
        source_revision=source_revision,
        llama_cpp_version=llama_cpp_version,
    )
    return BaseGgufBuildResult(
        gguf_path=out_gguf,
        sha256_hex=provenance["sha256_hex"],
        size_bytes=provenance["size_bytes"],
        provenance=provenance,
    )


def publish_base_gguf(
    result: BaseGgufBuildResult,
    *,
    object_store: Any,
    bucket: str,
    key: str,
) -> dict[str, Any]:
    """Upload the FP16 base GGUF to R2 and verify the stored object round-trips.

    The local sha256 is written as object metadata and read back via ``head`` so
    a truncated/corrupted upload is caught here (fail closed) rather than handed
    to agents. ``object_store`` is the seam (``put_object`` / ``head_object``);
    the deploy path wraps boto3, tests inject an in-memory fake.
    """
    object_store.put_object(
        bucket=bucket,
        key=key,
        file_path=result.gguf_path,
        metadata={"sha256": result.sha256_hex},
    )
    head = object_store.head_object(bucket=bucket, key=key)
    uploaded_sha = (head.get("metadata") or {}).get("sha256")
    if uploaded_sha != result.sha256_hex:
        raise BaseGgufBuildError(
            f"uploaded object sha256 mismatch for s3://{bucket}/{key}: "
            f"local={result.sha256_hex} uploaded={uploaded_sha}",
        )
    uploaded_size = head.get("size")
    if uploaded_size is not None and uploaded_size != result.size_bytes:
        raise BaseGgufBuildError(
            f"uploaded object size mismatch for s3://{bucket}/{key}: "
            f"local={result.size_bytes} uploaded={uploaded_size}",
        )
    return {
        "bucket": bucket,
        "key": key,
        "sha256_hex": result.sha256_hex,
        "size_bytes": result.size_bytes,
    }


class R2ObjectStore:
    """boto3-backed object store seam for Cloudflare R2 (S3-compatible).

    Lazy-imports boto3 so the rest of the module (provenance, build, key,
    reconciliation) stays dependency-light and unit-testable. Credentials and
    endpoint come from env, matching the engine's R2 wiring: ``R2_ENDPOINT``,
    ``R2_ACCESS_KEY_ID``, ``R2_SECRET_ACCESS_KEY`` (``R2_BUCKET`` is passed in).
    """

    def __init__(
        self,
        *,
        endpoint: str | None,
        access_key_id: str,
        secret_access_key: str,
        region: str = "auto",
    ):
        import boto3  # lazy

        self._client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            region_name=region,
        )

    def put_object(self, *, bucket, key, file_path, metadata):
        # ExtraArgs.Metadata rides along (incl. multipart) for large GGUFs.
        self._client.upload_file(
            str(file_path),
            bucket,
            key,
            ExtraArgs={"Metadata": {k: str(v) for k, v in metadata.items()}},
        )

    def head_object(self, *, bucket, key):
        head = self._client.head_object(Bucket=bucket, Key=key)
        return {
            "size": head.get("ContentLength"),
            "metadata": head.get("Metadata") or {},  # boto3 lowercases keys
        }
