"""Assemble a tiny-chat (``ollama-gguf-local``) submission bundle in the kit.

An external agent builds a real GGUF on its own compute (see
``gguf_build_pipeline``), then ships it as a bundle the engine admits via
``assertOllamaGgufBundleValid`` (``engine/src/v2/protocol/submissions/manifest.ts``):
the primary ``.gguf`` plus an Ollama ``Modelfile``, ``provenance.json``, and
``checksums.json``, declared in a manifest envelope.

This mirrors the engine-side producer ``engine/src/v2/verifier/tiny-chat-bundle.ts``
so the bytes an agent uploads and the manifest it declares agree with what the
engine expects. We reuse :class:`~codepit_optimizer.bundle.BundleFile` so the
orchestrator's presigned-upload path (`_drive_uploads`) consumes this bundle
unchanged.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .bundle import BundleFile

PRIMARY_GGUF_LOGICAL_NAME = "tiny-chat.gguf"
OLLAMA_GGUF_LOCAL_ARTIFACT_LANE = "ollama-gguf-local"
_MODEL_CLASS = "chat-causal-small"
_ENVIRONMENT_FAMILY = "local-ollama"
_RUNTIME = "ollama"
_PROMPT_TEMPLATE = "{{ .Prompt }}"
DEFAULT_TINY_CHAT_SYSTEM_PROMPT = (
    "Answer exactly what the user asks in one concise sentence. "
    "For rewrite/email tasks, return only the final requested text."
)


def _sha256_hex(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _bundle_file(logical_name: str, role: str, media_type: str, content: bytes) -> BundleFile:
    return BundleFile(
        logical_name=logical_name,
        role=role,
        media_type=media_type,
        size_bytes=len(content),
        sha256_hex=_sha256_hex(content),
        content=content,
    )


def build_tiny_chat_bundle_files(
    *,
    gguf_bytes: bytes,
    base_model_ref: str,
    quantization_profile: str,
) -> list[BundleFile]:
    """Build the four declared files for an ``ollama-gguf-local`` submission.

    The GGUF binary is shipped verbatim; the Modelfile, provenance, and
    checksums are synthesized so the bundle is self-describing. ``checksums.json``
    covers the GGUF and the other metadata (it cannot checksum itself).
    """
    modelfile = "\n".join(
        [
            f"FROM ./{PRIMARY_GGUF_LOGICAL_NAME}",
            f'TEMPLATE "{_PROMPT_TEMPLATE}"',
            f'SYSTEM "{DEFAULT_TINY_CHAT_SYSTEM_PROMPT}"',
            "",
        ],
    ).encode("utf-8")
    provenance = _json_bytes(
        {
            "schema": "codepit.tiny_chat.provenance.v1",
            "base_model_ref": base_model_ref,
            "quantization_profile": quantization_profile,
        },
    )
    checksums = _json_bytes(
        {
            "schema": "codepit.artifact_checksums.v1",
            "files": {
                PRIMARY_GGUF_LOGICAL_NAME: _sha256_hex(gguf_bytes),
                "Modelfile": _sha256_hex(modelfile),
                "provenance.json": _sha256_hex(provenance),
            },
        },
    )

    return [
        _bundle_file(PRIMARY_GGUF_LOGICAL_NAME, "primary_model", "application/x-gguf", gguf_bytes),
        _bundle_file("Modelfile", "metadata", "text/plain", modelfile),
        _bundle_file("provenance.json", "metadata", "application/json", provenance),
        _bundle_file("checksums.json", "metadata", "application/json", checksums),
    ]


def to_tiny_chat_manifest_envelope(
    files: list[BundleFile],
    *,
    benchmark_target_version: str,
    base_model_ref: str,
    quantization_profile: str,
    optimization_methods: list[str],
    source_model_revision: str = "main",
) -> dict[str, Any]:
    """Produce the manifest envelope the engine expects in ``POST /v1/submissions``.

    ``source_model.revision`` is declared so the engine can match a baseline
    submission against the challenge's configured ``baseline_reference`` (the
    official-baseline gate).
    """
    supporting = [
        file.logical_name for file in files if file.logical_name != PRIMARY_GGUF_LOGICAL_NAME
    ]
    return {
        "schema_version": benchmark_target_version,
        "artifact_lane": OLLAMA_GGUF_LOCAL_ARTIFACT_LANE,
        "model_class": _MODEL_CLASS,
        "benchmark_target_ref": benchmark_target_version,
        "source_model": {
            "identifier": base_model_ref,
            "revision": source_model_revision,
            "family": _MODEL_CLASS,
        },
        "runtime_target": {
            "environment_family": _ENVIRONMENT_FAMILY,
            "runtime": _RUNTIME,
            "constraints": {
                "quantization_profile": quantization_profile,
                "prompt_template": _PROMPT_TEMPLATE,
            },
        },
        "optimization": {"methods": optimization_methods},
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
            "primary_model_logical_name": PRIMARY_GGUF_LOGICAL_NAME,
            "supporting_files": supporting,
        },
    }
