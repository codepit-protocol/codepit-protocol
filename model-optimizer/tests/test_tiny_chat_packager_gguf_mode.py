import json
from pathlib import Path

from codepit_optimizer.tiny_chat_packager import train_and_package_tiny_chat

_MODELBOOK = {
    "modelbook_id": "mb_1",
    "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
    "artifact_lane": "ollama-gguf-local",
}


def _primary_model_path(package) -> Path:
    primary = next(f for f in package.files if f.role == "primary_model")
    return primary.path


def test_default_path_writes_the_json_gguf_fixture(tmp_path):
    package = train_and_package_tiny_chat(
        modelbook=_MODELBOOK,
        context={},
        training_run_id="run_default",
        recipe_kind="dynamic-int8",
        hyperparameters={},
        quantization_profile="q4_k_m",
        output_root=tmp_path,
    )

    # the fixture is a fake GGUF: magic header + embedded JSON body
    gguf_bytes = _primary_model_path(package).read_bytes()
    assert gguf_bytes[:4] == b"GGUF"
    assert b"codepit.tiny_chat.gguf_fixture.v1" in gguf_bytes
    assert "gguf_build" not in package.provenance


def test_injected_builder_produces_a_real_gguf_and_records_build_provenance(tmp_path):
    captured: dict[str, object] = {}

    def stub_build(*, base_model_ref, quantization_profile, out_gguf: Path):
        captured["base_model_ref"] = base_model_ref
        captured["quantization_profile"] = quantization_profile
        out_gguf.write_bytes(b"GGUF" + b"\x00" * 16)
        return {
            "schema": "codepit.tiny_chat.gguf_build.v1",
            "quant_type": "Q4_K_M",
            "base_model_ref": base_model_ref,
        }

    package = train_and_package_tiny_chat(
        modelbook=_MODELBOOK,
        context={},
        training_run_id="run_real",
        recipe_kind="dynamic-int8",
        hyperparameters={},
        quantization_profile="q4_k_m",
        output_root=tmp_path,
        gguf_build=stub_build,
    )

    # the builder was driven with the resolved base model + requested profile
    assert captured["base_model_ref"] == "hf://Qwen/Qwen2.5-0.5B-Instruct"
    assert captured["quantization_profile"] == "q4_k_m"

    # the emitted primary model is a real GGUF binary, not the JSON fixture
    gguf = _primary_model_path(package)
    assert gguf.read_bytes()[:4] == b"GGUF"

    # the build provenance is recorded, and the checksum covers the real binary
    assert package.provenance["gguf_build"]["schema"] == "codepit.tiny_chat.gguf_build.v1"
    checksum_file = next(f for f in package.files if f.logical_name == "checksums.json")
    checksums = json.loads(checksum_file.path.read_text())
    assert checksums["files"][gguf.name] == package.provenance["checksums"][gguf.name]
