"""Tests for the hosted-base quantize-only GGUF builder (slice C / #274).

This is the core ergonomics fix: a fresh external agent needs only the
``llama-quantize`` binary (one ``brew install``), not the full convert toolchain,
torch, or an HF download. The builder downloads the hosted FP16 base, then runs
``llama-quantize`` only. Subprocess + download are injected as seams.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from codepit_optimizer.gguf_build_pipeline import (
    GgufBuildError,
    build_hosted_base_quantized_gguf,
    resolve_quantize_bin,
)


def _quantize_runner(out_bytes: bytes = b"GGUF\x00quantized"):
    """Fake llama-quantize: writes prearranged bytes to the out path (argv[2])."""

    def runner(command, check=False):
        # build_quantize_command -> [bin, src, out, quant_type]
        Path(command[2]).write_bytes(out_bytes)
        return None

    return runner


def _base_provider(base_bytes: bytes = b"GGUF\x00fp16-base"):
    """Fake hosted-base resolver: writes the FP16 base to the requested path."""

    def provider(dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base_bytes)
        return dest

    return provider


def test_resolve_quantize_bin_prefers_explicit_env():
    env = {"CODEPIT_GGUF_QUANTIZE_BIN": "/opt/custom/llama-quantize"}
    assert resolve_quantize_bin(env=env, which=lambda _name: None) == "/opt/custom/llama-quantize"


def test_resolve_quantize_bin_auto_detects_on_path_when_env_unset():
    found = resolve_quantize_bin(env={}, which=lambda name: f"/usr/local/bin/{name}")
    assert found == "/usr/local/bin/llama-quantize"


def test_resolve_quantize_bin_fails_closed_with_actionable_message():
    with pytest.raises(GgufBuildError) as exc:
        resolve_quantize_bin(env={}, which=lambda _name: None)
    # actionable: tells the agent the one-line fix
    assert "brew install llama.cpp" in str(exc.value)


def test_quantize_only_builds_from_hosted_base_and_records_provenance(tmp_path):
    out = tmp_path / "tiny-chat.gguf"
    provenance = build_hosted_base_quantized_gguf(
        base_provider=_base_provider(),
        quantize_bin="/usr/local/bin/llama-quantize",
        quantization_profile="q4_k_m",
        out_gguf=out,
        runner=_quantize_runner(),
    )

    assert out.read_bytes()[:4] == b"GGUF"
    assert provenance["quantization_profile"] == "q4_k_m"
    assert provenance["quant_type"] == "Q4_K_M"
    assert provenance["method"] == "quantize-only-from-hosted-base"
    # provenance records both the base and the produced artifact hashes
    assert len(provenance["base_sha256"]) == 64
    assert len(provenance["gguf_sha256"]) == 64
    assert provenance["base_sha256"] != provenance["gguf_sha256"]


def test_quantize_only_rejects_f16_noop(tmp_path):
    # Quantizing an FP16 base to F16 is a byte-identical no-op — not a real
    # optimization. Must be rejected so the base can't be resubmitted as work.
    with pytest.raises(GgufBuildError):
        build_hosted_base_quantized_gguf(
            base_provider=_base_provider(),
            quantize_bin="/usr/local/bin/llama-quantize",
            quantization_profile="f16",
            out_gguf=tmp_path / "out.gguf",
            runner=_quantize_runner(),
        )


def test_quantize_only_rejects_profile_outside_tiny_chat_allowlist(tmp_path):
    with pytest.raises(GgufBuildError):
        build_hosted_base_quantized_gguf(
            base_provider=_base_provider(),
            quantize_bin="/usr/local/bin/llama-quantize",
            quantization_profile="q2_k",  # valid llama type, not allowed on the lane
            out_gguf=tmp_path / "out.gguf",
            runner=_quantize_runner(),
        )


def test_quantize_only_fails_closed_when_output_not_gguf(tmp_path):
    with pytest.raises(GgufBuildError):
        build_hosted_base_quantized_gguf(
            base_provider=_base_provider(),
            quantize_bin="/usr/local/bin/llama-quantize",
            quantization_profile="q4_k_m",
            out_gguf=tmp_path / "out.gguf",
            runner=_quantize_runner(out_bytes=b"NOT-A-GGUF"),
        )


# --- orchestrator-level wiring (slice C / #274) ---


def test_default_builder_falls_back_to_hosted_base(monkeypatch, tmp_path):
    """With no CODEPIT_GGUF_* toolchain env, the default builder is the
    hosted-base quantize-only path (not None) — the zero-config lightweight path.
    """
    from codepit_optimizer import orchestrator as o

    for var in (
        "CODEPIT_GGUF_CONVERT_SCRIPT",
        "CODEPIT_GGUF_QUANTIZE_BIN",
        "CODEPIT_GGUF_BASE_MODEL_DIR",
    ):
        monkeypatch.delenv(var, raising=False)

    builder = o._resolve_default_gguf_builder()
    assert builder is not None


def test_hosted_base_cache_path_is_per_url(tmp_path):
    from codepit_optimizer import orchestrator as o

    a = o._hosted_base_cache_path(tmp_path / "work", "https://x/base-a.gguf")
    b = o._hosted_base_cache_path(tmp_path / "work", "https://x/base-b.gguf")
    assert a != b  # distinct URLs must not collide on one cache filename


def test_default_base_fp16_url_is_distinct_from_finished_gguf_env():
    from codepit_optimizer import orchestrator as o

    # The footgun guard: base-FP16 (download+quantize) vs finished-GGUF
    # (submit-as-is) must be different env vars.
    assert o.TINY_CHAT_BASE_FP16_URL_ENV != o.TINY_CHAT_GGUF_URL_ENV
    assert o.DEFAULT_TINY_CHAT_BASE_FP16_URL.startswith("https://")
    assert "base-models/tiny-chat/fp16.gguf" in o.DEFAULT_TINY_CHAT_BASE_FP16_URL
