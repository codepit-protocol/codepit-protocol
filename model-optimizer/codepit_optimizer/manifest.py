from __future__ import annotations

from collections.abc import Iterable
from hashlib import sha256
from pathlib import Path
from typing import Any

FileTuple = tuple[str, str, str, int, str]


def build_manifest(
    *,
    benchmark_target_version: str,
    source_model_identifier: str,
    source_model_revision: str,
    files: Iterable[FileTuple],
    optimization_methods: list[str],
    optimization_notes: str,
    gitlawb: dict[str, str] | None = None,
) -> dict:
    file_declarations = [
        {
            "logical_name": logical_name,
            "role": role,
            "media_type": media_type,
            "size_bytes": size_bytes,
            "sha256": digest,
        }
        for logical_name, role, media_type, size_bytes, digest in files
    ]
    supporting = [
        item["logical_name"]
        for item in file_declarations
        if item["logical_name"] != "model.onnx"
    ]
    manifest: dict[str, Any] = {
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
        "files": file_declarations,
        "entrypoint": {
            "primary_model_logical_name": "model.onnx",
            "supporting_files": supporting,
        },
    }
    if gitlawb is not None:
        manifest["provenance"] = {
            "gitlawb": {
                "did": gitlawb["did"],
                "repo": gitlawb["repo"],
                "commit": gitlawb["commit"],
                "profile_url": gitlawb["profile_url"],
            },
        }
    return manifest


def build_file_declarations(bundle_dir: Path) -> list[FileTuple]:
    files: list[FileTuple] = []
    for path in sorted(bundle_dir.iterdir(), key=lambda item: item.name):
        if not path.is_file():
            continue
        role = _role_for_file(path.name)
        if role is None:
            continue
        content = path.read_bytes()
        files.append(
            (
                path.name,
                role,
                _media_type_for_file(path.name),
                len(content),
                sha256(content).hexdigest(),
            )
        )

    files.sort(key=lambda item: (0 if item[0] == "model.onnx" else 1, item[0]))
    return files


def _role_for_file(name: str) -> str | None:
    if name == "model.onnx":
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
