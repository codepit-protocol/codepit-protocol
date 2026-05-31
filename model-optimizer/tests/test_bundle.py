"""Bundle loader tests."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from codepit_optimizer.bundle import (
    BundleLoadError,
    load_bundle,
    to_manifest_envelope,
)


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def test_load_bundle_orders_primary_first_and_computes_hashes(tmp_path: Path) -> None:
    primary = b"\x00onnx-bytes"
    config_bytes = b'{"hidden":1}'
    tokenizer = b'{"vocab":[]}'
    _write(tmp_path / "model.onnx", primary)
    _write(tmp_path / "config.json", config_bytes)
    _write(tmp_path / "tokenizer.json", tokenizer)
    _write(tmp_path / "ignored.log", b"unused")

    files = load_bundle(tmp_path)

    assert [file.logical_name for file in files] == ["model.onnx", "config.json", "tokenizer.json"]
    assert files[0].role == "primary_model"
    assert files[0].size_bytes == len(primary)
    assert files[0].sha256_hex == hashlib.sha256(primary).hexdigest()
    assert files[0].media_type == "application/onnx"
    assert files[0].content == primary


def test_load_bundle_requires_primary_model(tmp_path: Path) -> None:
    _write(tmp_path / "config.json", b"{}")
    with pytest.raises(BundleLoadError):
        load_bundle(tmp_path)


def test_load_bundle_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(BundleLoadError):
        load_bundle(tmp_path / "does-not-exist")


def test_to_manifest_envelope_emits_expected_shape(tmp_path: Path) -> None:
    _write(tmp_path / "model.onnx", b"\x00")
    _write(tmp_path / "config.json", b"{}")
    files = load_bundle(tmp_path)

    envelope = to_manifest_envelope(
        files,
        benchmark_target_version="0.1.0",
        source_model_identifier="hf://org/model",
        source_model_revision="main",
        optimization_methods=["graph-optimization"],
        optimization_notes="recipe=graph-optimization",
    )

    assert envelope["artifact_lane"] == "onnx-browser-webgpu"
    assert envelope["model_class"] == "encoder-text-small"
    assert envelope["benchmark_target_ref"] == "0.1.0"
    assert envelope["source_model"] == {
        "identifier": "hf://org/model",
        "revision": "main",
        "family": "encoder-text-small",
    }
    assert envelope["entrypoint"]["primary_model_logical_name"] == "model.onnx"
    assert envelope["entrypoint"]["supporting_files"] == ["config.json"]
    assert {file["logical_name"] for file in envelope["files"]} == {"model.onnx", "config.json"}
