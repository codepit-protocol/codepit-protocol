import hashlib
from pathlib import Path
from subprocess import CalledProcessError, CompletedProcess

import pytest

from codepit_optimizer.gguf_build_pipeline import (
    GgufBuildError,
    GgufToolchain,
    build_convert_command,
    build_gguf,
    build_quantize_command,
    make_env_gguf_builder,
    resolve_quant_type,
    validate_gguf_magic,
)

_GGUF_ENV_KEYS = (
    "CODEPIT_GGUF_CONVERT_SCRIPT",
    "CODEPIT_GGUF_QUANTIZE_BIN",
    "CODEPIT_GGUF_BASE_MODEL_DIR",
    "CODEPIT_GGUF_PYTHON_BIN",
)

GGUF_MAGIC = b"GGUF"


def _toolchain() -> GgufToolchain:
    return GgufToolchain(
        convert_script="/opt/llama.cpp/convert_hf_to_gguf.py",
        quantize_bin="/opt/llama.cpp/llama-quantize",
        python_bin="python3",
    )


def test_resolve_quant_type_maps_known_profiles_and_rejects_unknown():
    assert resolve_quant_type("q4_k_m") == "Q4_K_M"
    assert resolve_quant_type("Q8_0") == "Q8_0"

    with pytest.raises(GgufBuildError) as err:
        resolve_quant_type("q3_made_up")
    assert "q3_made_up" in str(err.value)


def test_build_convert_command_targets_the_outfile_in_gguf_format(tmp_path):
    toolchain = _toolchain()
    model_dir = tmp_path / "Qwen2.5-0.5B-Instruct"
    out = tmp_path / "model-f16.gguf"

    command = build_convert_command(toolchain, model_dir, out, out_type="f16")

    assert command[0] == "python3"
    assert command[1] == "/opt/llama.cpp/convert_hf_to_gguf.py"
    assert str(model_dir) in command
    assert command[command.index("--outfile") + 1] == str(out)
    assert command[command.index("--outtype") + 1] == "f16"


def test_build_quantize_command_passes_source_output_and_uppercased_type(tmp_path):
    toolchain = _toolchain()
    src = tmp_path / "model-f16.gguf"
    out = tmp_path / "model-q4_k_m.gguf"

    command = build_quantize_command(toolchain, src, out, "Q4_K_M")

    assert command == ["/opt/llama.cpp/llama-quantize", str(src), str(out), "Q4_K_M"]


def test_validate_gguf_magic_accepts_real_header_and_rejects_otherwise(tmp_path):
    good = tmp_path / "good.gguf"
    good.write_bytes(GGUF_MAGIC + b"\x00" * 16)
    validate_gguf_magic(good)  # does not raise

    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOTGGUF")
    with pytest.raises(GgufBuildError):
        validate_gguf_magic(bad)


def test_build_gguf_runs_convert_then_quantize_and_returns_a_checksummed_result(tmp_path):
    toolchain = _toolchain()
    calls: list[list[str]] = []

    def runner(command, check=True):
        calls.append(command)
        if toolchain.convert_script in command:
            out = Path(command[command.index("--outfile") + 1])
            out.write_bytes(GGUF_MAGIC + b"\x00" * 16)
        else:
            out = Path(command[-2])
            out.write_bytes(GGUF_MAGIC + b"\x00" * 32)
        return CompletedProcess(command, 0)

    result = build_gguf(
        toolchain=toolchain,
        base_model_ref="hf://Qwen/Qwen2.5-0.5B-Instruct",
        base_model_dir=tmp_path / "base",
        quantization_profile="q4_k_m",
        work_dir=tmp_path / "out",
        runner=runner,
    )

    # convert ran before quantize
    assert toolchain.convert_script in calls[0]
    assert toolchain.quantize_bin in calls[1]

    # a real GGUF was produced, checksummed, and described in provenance
    assert result.gguf_path.exists()
    assert result.quant_type == "Q4_K_M"
    assert result.size_bytes == result.gguf_path.stat().st_size
    assert result.sha256_hex == hashlib.sha256(result.gguf_path.read_bytes()).hexdigest()
    assert result.provenance["base_model_ref"] == "hf://Qwen/Qwen2.5-0.5B-Instruct"
    assert result.provenance["quantization_profile"] == "q4_k_m"
    assert result.provenance["quant_type"] == "Q4_K_M"


def test_build_gguf_rejects_an_unknown_profile_without_running_any_tool(tmp_path):
    calls: list[list[str]] = []

    def runner(command, check=True):
        calls.append(command)
        return CompletedProcess(command, 0)

    with pytest.raises(GgufBuildError):
        build_gguf(
            toolchain=_toolchain(),
            base_model_ref="hf://Qwen/Qwen2.5-0.5B-Instruct",
            base_model_dir=tmp_path / "base",
            quantization_profile="totally-made-up",
            work_dir=tmp_path / "out",
            runner=runner,
        )
    assert calls == []


def test_build_gguf_fails_closed_when_a_tool_errors(tmp_path):
    def runner(command, check=True):
        raise CalledProcessError(1, command)

    with pytest.raises(GgufBuildError):
        build_gguf(
            toolchain=_toolchain(),
            base_model_ref="hf://Qwen/Qwen2.5-0.5B-Instruct",
            base_model_dir=tmp_path / "base",
            quantization_profile="q4_k_m",
            work_dir=tmp_path / "out",
            runner=runner,
        )


def test_make_env_gguf_builder_returns_none_when_toolchain_is_unconfigured(monkeypatch):
    for key in _GGUF_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    assert make_env_gguf_builder() is None


def test_make_env_gguf_builder_returns_a_builder_when_toolchain_is_configured(monkeypatch, tmp_path):
    monkeypatch.setenv("CODEPIT_GGUF_CONVERT_SCRIPT", "/opt/llama.cpp/convert_hf_to_gguf.py")
    monkeypatch.setenv("CODEPIT_GGUF_QUANTIZE_BIN", "/opt/llama.cpp/llama-quantize")
    monkeypatch.setenv("CODEPIT_GGUF_BASE_MODEL_DIR", str(tmp_path / "base"))

    builder = make_env_gguf_builder()
    assert builder is not None
    assert callable(builder)


def test_build_gguf_fails_closed_when_the_output_is_not_a_real_gguf(tmp_path):
    toolchain = _toolchain()

    def runner(command, check=True):
        if toolchain.convert_script in command:
            Path(command[command.index("--outfile") + 1]).write_bytes(GGUF_MAGIC + b"\x00" * 16)
        else:
            Path(command[-2]).write_bytes(b"NOT-A-GGUF")
        return CompletedProcess(command, 0)

    with pytest.raises(GgufBuildError):
        build_gguf(
            toolchain=toolchain,
            base_model_ref="hf://Qwen/Qwen2.5-0.5B-Instruct",
            base_model_dir=tmp_path / "base",
            quantization_profile="q4_k_m",
            work_dir=tmp_path / "out",
            runner=runner,
        )
