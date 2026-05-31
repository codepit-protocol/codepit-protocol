from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

from codepit_optimizer.bundle import BundleFile
from codepit_optimizer.modelbook_submission import _drive_uploads as drive_modelbook_uploads
from codepit_optimizer.orchestrator import _drive_uploads as drive_orchestrator_uploads
from codepit_optimizer.tiny_chat_packager import TinyChatArtifactFile, TinyChatArtifactPackage


class RecordingClient:
    def __init__(self) -> None:
        self.puts: list[tuple[str, bytes, str]] = []

    def put_bytes(self, upload_url: str, content: bytes, content_type: str) -> None:
        self.puts.append((upload_url, content, content_type))


def valid_expires_at() -> str:
    return (datetime.now(timezone.utc) + timedelta(minutes=15)).isoformat().replace("+00:00", "Z")


def test_modelbook_submission_skips_already_uploaded_files(tmp_path):
    content = b"gguf-bytes"
    path = tmp_path / "tiny-chat.gguf"
    path.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    package = TinyChatArtifactPackage(
        output_dir=tmp_path,
        primary_artifact_ref=path.as_uri(),
        adapter_ref=path.as_uri(),
        merged_model_ref=path.as_uri(),
        gguf_ref=path.as_uri(),
        modelfile_ref=path.as_uri(),
        checksum_ref=path.as_uri(),
        provenance={},
        dataset_shard_ids=[],
        progress_events=[],
        files=[
            TinyChatArtifactFile(
                logical_name="tiny-chat.gguf",
                role="primary_model",
                media_type="application/x-gguf",
                path=path,
                size_bytes=len(content),
                sha256_hex=sha,
            )
        ],
    )
    client = RecordingClient()

    drive_modelbook_uploads(
        client,
        {
            "upload_orchestration": {
                "kind": "presigned-urls",
                "expires_at": valid_expires_at(),
                "files": [
                    {
                        "logical_name": "tiny-chat.gguf",
                        "role": "primary_model",
                        "media_type": "application/x-gguf",
                        "size_bytes": len(content),
                        "sha256": sha,
                        "object_key": f"artifacts/sha256/{sha}/00-tiny-chat.gguf",
                        "already_uploaded": True,
                        "upload_url": "http://uploads.test/tiny-chat.gguf",
                    }
                ],
            }
        },
        package,
    )

    assert client.puts == []


def test_orchestrator_skips_already_uploaded_files():
    content = b"onnx-bytes"
    sha = hashlib.sha256(content).hexdigest()
    bundle_file = BundleFile(
        logical_name="model.onnx",
        role="primary_model",
        media_type="application/onnx",
        size_bytes=len(content),
        sha256_hex=sha,
        content=content,
    )
    client = RecordingClient()

    drive_orchestrator_uploads(
        client,
        {
            "upload_orchestration": {
                "kind": "presigned-urls",
                "expires_at": valid_expires_at(),
                "files": [
                    {
                        "logical_name": "model.onnx",
                        "role": "primary_model",
                        "media_type": "application/onnx",
                        "size_bytes": len(content),
                        "sha256": sha,
                        "object_key": f"artifacts/sha256/{sha}/00-model.onnx",
                        "already_uploaded": True,
                        "upload_url": "http://uploads.test/model.onnx",
                    }
                ],
            }
        },
        [bundle_file],
    )

    assert client.puts == []
