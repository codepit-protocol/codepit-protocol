"""Load an optimizer-produced candidate bundle from disk.

A bundle is one directory of files produced by a recipe (ONNX export +
optional optimizer + optional quantizer). The orchestrator turns one of
those directories into:
- a manifest declaration (file role, size, sha256) for ``POST /v1/submissions``
- the file bytes themselves for the presigned upload step

We deliberately model both together so a candidate is one immutable record
and the manifest hash agrees with what we actually upload.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

PRIMARY_MODEL_LOGICAL_NAME = "model.onnx"


@dataclass(frozen=True)
class BundleFile:
    logical_name: str
    role: str
    media_type: str
    size_bytes: int
    sha256_hex: str
    content: bytes


class BundleLoadError(RuntimeError):
    """Raised when a candidate directory is missing the primary model file or
    contains files with no recognized role.
    """


def load_bundle(bundle_dir: Path) -> list[BundleFile]:
    """Read every recognized file under ``bundle_dir`` and return BundleFiles.

    Files with no recognized role are silently skipped — recipes sometimes
    emit auxiliary artifacts (logs, metric dumps) we don't want to ship.
    The primary ``model.onnx`` is required; missing it is a hard error.
    """
    if not bundle_dir.is_dir():
        raise BundleLoadError(f"bundle directory does not exist: {bundle_dir}")

    files: list[BundleFile] = []
    for path in sorted(bundle_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        role = _role_for_file(path.name)
        if role is None:
            continue
        content = path.read_bytes()
        files.append(
            BundleFile(
                logical_name=path.name,
                role=role,
                media_type=_media_type_for_file(path.name),
                size_bytes=len(content),
                sha256_hex=hashlib.sha256(content).hexdigest(),
                content=content,
            ),
        )

    if not any(file.logical_name == PRIMARY_MODEL_LOGICAL_NAME for file in files):
        raise BundleLoadError(
            f"bundle {bundle_dir} is missing required primary model: {PRIMARY_MODEL_LOGICAL_NAME}",
        )

    files.sort(key=lambda item: (0 if item.logical_name == PRIMARY_MODEL_LOGICAL_NAME else 1, item.logical_name))
    return files


def to_manifest_envelope(
    files: list[BundleFile],
    *,
    benchmark_target_version: str,
    source_model_identifier: str,
    source_model_revision: str,
    optimization_methods: list[str],
    optimization_notes: str,
) -> dict:
    """Produce the V2 manifest envelope the engine expects in POST /v1/submissions."""
    supporting = [
        file.logical_name
        for file in files
        if file.logical_name != PRIMARY_MODEL_LOGICAL_NAME
    ]
    return {
        "schema_version": benchmark_target_version,
        "artifact_lane": "onnx-browser-webgpu",
        "model_class": "encoder-text-small",
        "benchmark_target_ref": benchmark_target_version,
        "source_model": {
            "identifier": source_model_identifier,
            "revision": source_model_revision,
            "family": "encoder-text-small",
        },
        "runtime_target": {
            "environment_family": "browser-webgpu",
            "runtime": "onnxruntime-web-webgpu",
        },
        "optimization": {
            "methods": optimization_methods,
            "notes": optimization_notes,
        },
        "files": [
            {
                "logical_name": file.logical_name,
                "role": file.role,
                "media_type": file.media_type,
                "size_bytes": file.size_bytes,
                "sha256": file.sha256_hex,
            }
            for file in files
        ],
        "entrypoint": {
            "primary_model_logical_name": PRIMARY_MODEL_LOGICAL_NAME,
            "supporting_files": supporting,
        },
    }


def _role_for_file(name: str) -> str | None:
    if name == PRIMARY_MODEL_LOGICAL_NAME:
        return "primary_model"
    if name == "model.onnx_data":
        return "model_data"
    if name == "config.json":
        return "config"
    if name in {"tokenizer.json", "tokenizer_config.json"}:
        return "tokenizer"
    if name in {"vocab.txt", "vocab.json", "merges.txt"}:
        return "vocab"
    if name.endswith(".json"):
        return "metadata"
    return None


def _media_type_for_file(name: str) -> str:
    if name.endswith(".onnx"):
        return "application/onnx"
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".txt"):
        return "text/plain"
    return "application/octet-stream"
