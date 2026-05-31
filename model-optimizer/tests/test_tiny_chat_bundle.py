"""Tiny-chat (ollama-gguf-local) submission bundle assembly in the kit.

The engine admits this lane via ``assertOllamaGgufBundleValid`` in
``engine/src/v2/protocol/submissions/manifest.ts``. These tests pin the kit's
producer output to that admission contract so an external agent's bundle is
accepted without round-tripping through the engine. They mirror the engine
producer ``engine/src/v2/verifier/tiny-chat-bundle.ts``.
"""

import json

from codepit_optimizer.tiny_chat_bundle import (
    DEFAULT_TINY_CHAT_SYSTEM_PROMPT,
    PRIMARY_GGUF_LOGICAL_NAME,
    build_tiny_chat_bundle_files,
    to_tiny_chat_manifest_envelope,
)

# A minimal but real GGUF binary: the magic header the engine/llama.cpp expect,
# plus padding. Admission checks the declared shape, not model internals.
_GGUF_BYTES = b"GGUF" + b"\x00" * 16
_BASE_MODEL_REF = "hf://Qwen/Qwen2.5-0.5B-Instruct"
_QUANT_PROFILE = "q4_k_m"
# Echoed back as manifest.benchmark_target_ref; the engine compares it to the
# challenge's benchmark_target_version at submission time.
_BENCHMARK_TARGET_VERSION = "ollama-gguf-local-v1"


def _by_logical_name(files):
    return {f.logical_name: f for f in files}


def test_assembles_the_four_required_gguf_bundle_files():
    files = build_tiny_chat_bundle_files(
        gguf_bytes=_GGUF_BYTES,
        base_model_ref=_BASE_MODEL_REF,
        quantization_profile=_QUANT_PROFILE,
    )

    by_name = _by_logical_name(files)
    assert set(by_name) == {
        PRIMARY_GGUF_LOGICAL_NAME,
        "Modelfile",
        "provenance.json",
        "checksums.json",
    }

    gguf = by_name[PRIMARY_GGUF_LOGICAL_NAME]
    assert gguf.role == "primary_model"
    assert gguf.media_type == "application/x-gguf"
    # the binary the agent built is shipped verbatim
    assert gguf.content == _GGUF_BYTES
    assert gguf.size_bytes == len(_GGUF_BYTES)

    assert by_name["Modelfile"].role == "metadata"
    assert by_name["Modelfile"].media_type == "text/plain"
    assert DEFAULT_TINY_CHAT_SYSTEM_PROMPT.encode("utf-8") in by_name["Modelfile"].content
    assert by_name["provenance.json"].media_type == "application/json"
    assert by_name["checksums.json"].media_type == "application/json"

    # checksums.json covers the real GGUF binary by its sha256
    checksums = json.loads(by_name["checksums.json"].content)
    assert checksums["files"][PRIMARY_GGUF_LOGICAL_NAME] == gguf.sha256_hex


def test_manifest_envelope_satisfies_ollama_gguf_admission():
    files = build_tiny_chat_bundle_files(
        gguf_bytes=_GGUF_BYTES,
        base_model_ref=_BASE_MODEL_REF,
        quantization_profile=_QUANT_PROFILE,
    )
    envelope = to_tiny_chat_manifest_envelope(
        files,
        benchmark_target_version=_BENCHMARK_TARGET_VERSION,
        base_model_ref=_BASE_MODEL_REF,
        source_model_revision="main",
        quantization_profile=_QUANT_PROFILE,
        optimization_methods=[_QUANT_PROFILE],
    )

    # lane + class + target the engine matches against the challenge snapshot
    assert envelope["artifact_lane"] == "ollama-gguf-local"
    assert envelope["model_class"] == "chat-causal-small"
    assert envelope["benchmark_target_ref"] == _BENCHMARK_TARGET_VERSION

    # source + optimization echo what the agent actually built. The revision is
    # required so the engine can match this submission against the challenge's
    # configured baseline_reference (the official-baseline gate).
    assert envelope["source_model"]["identifier"] == _BASE_MODEL_REF
    assert envelope["source_model"]["revision"] == "main"
    assert envelope["source_model"]["family"] == "chat-causal-small"
    assert envelope["optimization"]["methods"] == [_QUANT_PROFILE]

    # runtime_target constraints required by assertOllamaGgufBundleValid
    runtime_target = envelope["runtime_target"]
    assert runtime_target["environment_family"] == "local-ollama"
    assert "ollama" in runtime_target["runtime"].lower()
    constraints = runtime_target["constraints"]
    assert constraints["quantization_profile"] == _QUANT_PROFILE
    assert constraints["prompt_template"].strip() != ""

    # exactly one primary_model, a .gguf, referenced by the entrypoint
    primary_models = [f for f in envelope["files"] if f["role"] == "primary_model"]
    assert len(primary_models) == 1
    assert primary_models[0]["logical_name"].lower().endswith(".gguf")
    assert envelope["entrypoint"]["primary_model_logical_name"] == PRIMARY_GGUF_LOGICAL_NAME

    # every required supporting file is declared and present in the bundle
    declared = {f["logical_name"] for f in envelope["files"]}
    for required in ("Modelfile", "provenance.json", "checksums.json"):
        assert required in declared
        assert required in envelope["entrypoint"]["supporting_files"]
