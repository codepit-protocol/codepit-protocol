"""Tests for the FP16 base GGUF build + R2 publish (slice A / #271).

The llama.cpp convert binary and the R2 object store are injected as seams so we
exercise provenance, GGUF-magic validation (fail-closed), the stable R2 key, and
sha256 upload reconciliation without the real toolchain or network.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from codepit_optimizer.base_gguf_build import (
    BaseGgufBuildError,
    build_base_gguf_fp16,
    compute_base_gguf_provenance,
    default_base_gguf_key,
    publish_base_gguf,
)


class _FakeObjectStore:
    """In-memory stand-in for the R2/S3 object store seam."""

    def __init__(self, *, corrupt: bool = False):
        self._objects: dict = {}
        self._corrupt = corrupt

    def put_object(self, *, bucket, key, file_path, metadata):
        stored_meta = dict(metadata)
        if self._corrupt:
            stored_meta["sha256"] = "0" * 64  # simulate a corrupted upload
        self._objects[(bucket, key)] = {
            "size": Path(file_path).stat().st_size,
            "metadata": stored_meta,
        }

    def head_object(self, *, bucket, key):
        obj = self._objects.get((bucket, key))
        if obj is None:
            raise KeyError(key)
        return obj


def _write_gguf(path: Path, body: bytes = b"\x00\x01\x02\x03") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + body)


def _convert_runner(output_bytes: bytes = b"GGUF\x00\x01"):
    """A fake convert subprocess that writes prearranged bytes to the --outfile."""

    def runner(command, check=False):
        gguf_paths = [Path(tok) for tok in command if str(tok).endswith(".gguf")]
        gguf_paths[-1].parent.mkdir(parents=True, exist_ok=True)
        gguf_paths[-1].write_bytes(output_bytes)
        return None

    return runner


def _build(tmp_path, runner=None):
    return build_base_gguf_fp16(
        base_model_dir=tmp_path / "model",
        out_gguf=tmp_path / "out" / "fp16.gguf",
        base_model_ref="hf://Qwen/Qwen2.5-0.5B-Instruct",
        source_revision="main",
        convert_script="convert_hf_to_gguf.py",
        llama_cpp_version="b9270",
        runner=runner or _convert_runner(),
    )


def test_compute_provenance_records_identity_and_hash(tmp_path):
    artifact = tmp_path / "fp16.gguf"
    _write_gguf(artifact, b"hello-world-bytes")

    prov = compute_base_gguf_provenance(
        artifact,
        base_model_ref="hf://Qwen/Qwen2.5-0.5B-Instruct",
        source_revision="main",
        llama_cpp_version="b9270",
    )

    assert prov["base_model_ref"] == "hf://Qwen/Qwen2.5-0.5B-Instruct"
    assert prov["source_revision"] == "main"
    assert prov["llama_cpp_version"] == "b9270"
    assert prov["artifact_kind"] == "gguf-fp16"
    assert prov["size_bytes"] == artifact.stat().st_size
    assert prov["sha256_hex"] == hashlib.sha256(artifact.read_bytes()).hexdigest()


def test_build_fp16_converts_validates_and_records_provenance(tmp_path):
    result = _build(tmp_path)

    assert result.gguf_path.exists()
    assert result.gguf_path.read_bytes()[:4] == b"GGUF"
    assert result.provenance["artifact_kind"] == "gguf-fp16"
    assert result.provenance["base_model_ref"] == "hf://Qwen/Qwen2.5-0.5B-Instruct"
    assert result.sha256_hex == result.provenance["sha256_hex"]


def test_build_fp16_fails_closed_on_non_gguf_output(tmp_path):
    with pytest.raises(BaseGgufBuildError):
        _build(tmp_path, runner=_convert_runner(output_bytes=b"NOT-A-GGUF"))


def test_default_base_gguf_key_is_stable_and_lane_scoped():
    key = default_base_gguf_key()
    assert key == "base-models/tiny-chat/qwen2.5-0.5b-instruct/fp16.gguf"
    assert default_base_gguf_key() == key  # deterministic, no timestamps/randomness


def test_publish_uploads_with_sha256_and_verifies_roundtrip(tmp_path):
    result = _build(tmp_path)
    store = _FakeObjectStore()

    summary = publish_base_gguf(
        result, object_store=store, bucket="codepit-artifacts", key=default_base_gguf_key()
    )

    assert summary["bucket"] == "codepit-artifacts"
    assert summary["key"] == default_base_gguf_key()
    assert summary["sha256_hex"] == result.sha256_hex
    head = store.head_object(bucket="codepit-artifacts", key=default_base_gguf_key())
    assert head["metadata"]["sha256"] == result.sha256_hex
    assert head["size"] == result.size_bytes


def test_publish_fails_closed_when_uploaded_object_mismatches(tmp_path):
    result = _build(tmp_path)
    store = _FakeObjectStore(corrupt=True)

    with pytest.raises(BaseGgufBuildError):
        publish_base_gguf(
            result, object_store=store, bucket="codepit-artifacts", key=default_base_gguf_key()
        )


# --- "test the actual fix": real subprocess + real boto3 adapter (no Python fakes) ---


def test_build_fp16_through_real_subprocess(tmp_path):
    """Drive the real ``subprocess.run`` seam: a stand-in convert script obeying
    the real CLI contract runs as an actual OS process — proving command
    assembly + exec + magic validation + provenance end-to-end.
    """
    convert_script = tmp_path / "fake_convert_hf_to_gguf.py"
    convert_script.write_text(
        "import sys, pathlib\n"
        "out = sys.argv[sys.argv.index('--outfile') + 1]\n"
        "assert sys.argv[sys.argv.index('--outtype') + 1] == 'f16'\n"
        "pathlib.Path(out).write_bytes(b'GGUF' + b'real-subprocess-bytes')\n"
    )
    (tmp_path / "model").mkdir()

    result = build_base_gguf_fp16(
        base_model_dir=tmp_path / "model",
        out_gguf=tmp_path / "out" / "fp16.gguf",
        base_model_ref="hf://Qwen/Qwen2.5-0.5B-Instruct",
        source_revision="main",
        convert_script=str(convert_script),
        llama_cpp_version="b9270",
        # no runner= -> real subprocess.run
    )

    assert result.gguf_path.read_bytes()[:4] == b"GGUF"
    assert result.provenance["sha256_hex"] == result.sha256_hex


def test_build_fp16_through_real_subprocess_fails_closed_on_bad_output(tmp_path):
    convert_script = tmp_path / "bad_convert.py"
    convert_script.write_text(
        "import sys, pathlib\n"
        "out = sys.argv[sys.argv.index('--outfile') + 1]\n"
        "pathlib.Path(out).write_bytes(b'NOT-A-GGUF')\n"
    )
    (tmp_path / "model").mkdir()

    with pytest.raises(BaseGgufBuildError):
        build_base_gguf_fp16(
            base_model_dir=tmp_path / "model",
            out_gguf=tmp_path / "out" / "fp16.gguf",
            base_model_ref="hf://Qwen/Qwen2.5-0.5B-Instruct",
            source_revision="main",
            convert_script=str(convert_script),
            llama_cpp_version="b9270",
        )


def test_publish_through_real_boto3_against_mock_s3(tmp_path):
    """Exercise the real boto3 R2 adapter (upload + sha256-metadata round-trip)
    against a mock S3. Skipped where moto isn't installed (not a kit dep).
    """
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws

    from codepit_optimizer.base_gguf_build import R2ObjectStore

    with mock_aws():
        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket="codepit-artifacts")
        result = _build(tmp_path)
        store = R2ObjectStore(endpoint=None, access_key_id="test", secret_access_key="test")
        summary = publish_base_gguf(
            result, object_store=store, bucket="codepit-artifacts", key=default_base_gguf_key()
        )
        head = store.head_object(bucket="codepit-artifacts", key=default_base_gguf_key())

    assert summary["sha256_hex"] == result.sha256_hex
    assert head["metadata"].get("sha256") == result.sha256_hex
    assert head["size"] == result.size_bytes
