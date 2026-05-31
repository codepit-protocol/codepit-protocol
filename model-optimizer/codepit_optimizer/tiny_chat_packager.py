"""Deterministic Tiny Chat trainer and packager for Modelbook MVP evidence.

This module intentionally stays dependency-light so CI can exercise the
non-stub path. It trains a small prompt-response adapter over a fixed Tiny Chat
fixture, packages the derived adapter into local artifacts, and records
checksums for every emitted file. The output is not a general-purpose LLM
trainer; it is the first reproducible tiny-chat training fixture required by
the Modelbook MVP gate.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

# Optional real GGUF build seam: (base_model_ref, quantization_profile, out_gguf)
# -> provenance dict, producing a real .gguf at out_gguf. When None, the
# deterministic JSON fixture is written instead (CI default).
GgufBuilder = Callable[..., dict[str, Any]]


class TinyChatPackagingError(RuntimeError):
    """Raised when training, export, or checksum generation cannot complete."""


@dataclass(frozen=True)
class TinyChatArtifactFile:
    logical_name: str
    role: str
    media_type: str
    path: Path
    size_bytes: int
    sha256_hex: str


@dataclass(frozen=True)
class TinyChatArtifactPackage:
    output_dir: Path
    primary_artifact_ref: str
    adapter_ref: str
    merged_model_ref: str
    gguf_ref: str
    modelfile_ref: str
    checksum_ref: str
    provenance: dict[str, Any]
    dataset_shard_ids: list[str]
    files: list[TinyChatArtifactFile]
    progress_events: list[dict[str, Any]]


_TINY_CHAT_EXAMPLES: tuple[tuple[str, str], ...] = (
    (
        "What is CodePit?",
        "CodePit is a benchmark arena where agents improve small models and the platform verifies the result.",
    ),
    (
        "What is my agent doing?",
        "Your agent is choosing a safe training recipe, packaging the result, and sending it for platform checks.",
    ),
    (
        "Can I trust the score?",
        "Only the platform verifier can confirm a score, so agent updates stay separate from verified results.",
    ),
    (
        "What does a Modelbook do?",
        "A Modelbook gives the agent the model, data rules, export target, and success checks for one training job.",
    ),
)


def train_and_package_tiny_chat(
    *,
    modelbook: Mapping[str, Any],
    context: Mapping[str, Any],
    training_run_id: str,
    recipe_kind: str,
    hyperparameters: Mapping[str, Any],
    quantization_profile: str,
    output_root: str | Path | None = None,
    gguf_build: GgufBuilder | None = None,
) -> TinyChatArtifactPackage:
    """Train the deterministic Tiny Chat fixture and write package artifacts.

    When ``gguf_build`` is provided, the primary model is produced as a real
    GGUF binary via that seam (and its provenance recorded); otherwise the
    deterministic JSON GGUF fixture is written.
    """

    modelbook_id = _require_text(modelbook, "modelbook_id")
    base_model_ref = str(
        modelbook.get("base_model_ref")
        or (context.get("modelbook") or {}).get("base_model_ref")
        or "hf://Qwen/Qwen2.5-0.5B-Instruct"
    )
    artifact_lane = str(modelbook.get("artifact_lane") or "ollama-gguf-local")
    if artifact_lane != "ollama-gguf-local":
        raise TinyChatPackagingError(
            f"unsupported artifact lane {artifact_lane!r}; expected 'ollama-gguf-local'",
        )

    output_dir = _resolve_output_dir(output_root, training_run_id)
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise TinyChatPackagingError(
            f"could not create tiny-chat artifact directory {output_dir}: {error}",
        ) from error

    dataset_shards = list(context.get("dataset_shards") or [])
    dataset_shard_ids = [
        str(shard.get("dataset_shard_id"))
        for shard in dataset_shards
        if isinstance(shard, Mapping) and shard.get("dataset_shard_id")
    ]

    adapter = _train_adapter(
        modelbook_id=modelbook_id,
        base_model_ref=base_model_ref,
        recipe_kind=recipe_kind,
        hyperparameters=hyperparameters,
    )
    merged_model = _merge_tiny_chat_model(
        base_model_ref=base_model_ref,
        adapter=adapter,
        quantization_profile=quantization_profile,
    )

    adapter_path = output_dir / "adapter.json"
    merged_path = output_dir / "merged-model.json"
    gguf_path = output_dir / f"tiny-chat-{_safe_name(quantization_profile)}.gguf"
    modelfile_path = output_dir / "Modelfile"
    provenance_path = output_dir / "provenance.json"
    checksum_path = output_dir / "checksums.json"

    try:
        _write_json(adapter_path, adapter)
        _write_json(merged_path, merged_model)
        if gguf_build is not None:
            gguf_build_provenance = gguf_build(
                base_model_ref=base_model_ref,
                quantization_profile=quantization_profile,
                out_gguf=gguf_path,
            )
        else:
            _write_gguf_fixture(
                gguf_path,
                {
                    "schema": "codepit.tiny_chat.gguf_fixture.v1",
                    "base_model_ref": base_model_ref,
                    "adapter_sha256": _sha256_file(adapter_path),
                    "merged_model_sha256": _sha256_file(merged_path),
                    "quantization_profile": quantization_profile,
                    "training_run_id": training_run_id,
                },
            )
            gguf_build_provenance = None
        _write_text(
            modelfile_path,
            "\n".join(
                [
                    f"FROM ./{gguf_path.name}",
                    'TEMPLATE "{{ .Prompt }}\\n{{ .Response }}"',
                    'SYSTEM "You are a concise Tiny Chat model improved by a CodePit agent."',
                    "",
                ]
            ),
        )
        provenance = {
            "schema": "codepit.tiny_chat.provenance.v1",
            "modelbook_id": modelbook_id,
            "training_run_id": training_run_id,
            "base_model_ref": base_model_ref,
            "artifact_lane": artifact_lane,
            "recipe_kind": recipe_kind,
            "hyperparameters": dict(hyperparameters),
            "quantization_profile": quantization_profile,
            "training_algorithm": "deterministic-token-frequency-adapter",
            "training_example_count": len(_TINY_CHAT_EXAMPLES),
            "dataset_shard_ids": dataset_shard_ids,
            "artifact_files": {
                "adapter": adapter_path.name,
                "merged_model": merged_path.name,
                "gguf": gguf_path.name,
                "modelfile": modelfile_path.name,
                "checksums": checksum_path.name,
            },
        }
        if gguf_build_provenance is not None:
            provenance["gguf_build"] = gguf_build_provenance
        _write_json(provenance_path, provenance)
        checksums = {
            "schema": "codepit.artifact_checksums.v1",
            "files": {
                adapter_path.name: _sha256_file(adapter_path),
                merged_path.name: _sha256_file(merged_path),
                gguf_path.name: _sha256_file(gguf_path),
                modelfile_path.name: _sha256_file(modelfile_path),
                provenance_path.name: _sha256_file(provenance_path),
            },
        }
        _write_json(checksum_path, checksums)
    except OSError as error:
        raise TinyChatPackagingError(
            f"tiny-chat artifact write failed in {output_dir}: {error}",
        ) from error

    provenance["checksums"] = checksums["files"]
    artifact_files = [
        _artifact_file(gguf_path, role="primary_model", media_type="application/x-gguf"),
        _artifact_file(adapter_path, role="model_data", media_type="application/json"),
        _artifact_file(merged_path, role="metadata", media_type="application/json"),
        _artifact_file(modelfile_path, role="metadata", media_type="text/plain"),
        _artifact_file(provenance_path, role="metadata", media_type="application/json"),
        _artifact_file(checksum_path, role="metadata", media_type="application/json"),
    ]
    return TinyChatArtifactPackage(
        output_dir=output_dir,
        primary_artifact_ref=_file_ref(gguf_path),
        adapter_ref=_file_ref(adapter_path),
        merged_model_ref=_file_ref(merged_path),
        gguf_ref=_file_ref(gguf_path),
        modelfile_ref=_file_ref(modelfile_path),
        checksum_ref=_file_ref(checksum_path),
        provenance=provenance,
        dataset_shard_ids=dataset_shard_ids,
        files=artifact_files,
        progress_events=[
            {
                "event_type": "training.dataset_prepared",
                "message": f"Prepared {len(_TINY_CHAT_EXAMPLES)} Tiny Chat examples.",
                "metadata": {
                    "example_count": len(_TINY_CHAT_EXAMPLES),
                    "dataset_shard_ids": dataset_shard_ids,
                },
            },
            {
                "event_type": "training.adapter_trained",
                "message": "Trained a deterministic Tiny Chat response adapter.",
                "metadata": {
                    "training_algorithm": "deterministic-token-frequency-adapter",
                    "recipe_kind": recipe_kind,
                },
            },
            {
                "event_type": "training.packaged",
                "message": "Packaged adapter, merged model, GGUF file, Modelfile, provenance, and checksums.",
                "metadata": {
                    "artifact_dir": str(output_dir),
                    "quantization_profile": quantization_profile,
                    "checksum_count": len(provenance["checksums"]),
                },
            },
        ],
    )


def _train_adapter(
    *,
    modelbook_id: str,
    base_model_ref: str,
    recipe_kind: str,
    hyperparameters: Mapping[str, Any],
) -> dict[str, Any]:
    prompt_tokens: dict[str, int] = {}
    response_tokens: dict[str, int] = {}
    exemplars: list[dict[str, str]] = []
    for prompt, response in _TINY_CHAT_EXAMPLES:
        exemplars.append({"prompt": prompt, "response": response})
        for token in _tokens(prompt):
            prompt_tokens[token] = prompt_tokens.get(token, 0) + 1
        for token in _tokens(response):
            response_tokens[token] = response_tokens.get(token, 0) + 1

    if not prompt_tokens or not response_tokens:
        raise TinyChatPackagingError("tiny-chat fixture produced no trainable tokens")

    return {
        "schema": "codepit.tiny_chat.adapter.v1",
        "modelbook_id": modelbook_id,
        "base_model_ref": base_model_ref,
        "recipe_kind": recipe_kind,
        "hyperparameters": dict(hyperparameters),
        "prompt_token_weights": dict(sorted(prompt_tokens.items())),
        "response_token_weights": dict(sorted(response_tokens.items())),
        "exemplars": exemplars,
    }


def _merge_tiny_chat_model(
    *,
    base_model_ref: str,
    adapter: Mapping[str, Any],
    quantization_profile: str,
) -> dict[str, Any]:
    top_response_tokens = sorted(
        (adapter.get("response_token_weights") or {}).items(),
        key=lambda item: (-int(item[1]), str(item[0])),
    )[:12]
    return {
        "schema": "codepit.tiny_chat.merged_model.v1",
        "base_model_ref": base_model_ref,
        "adapter_schema": adapter.get("schema"),
        "quantization_profile": quantization_profile,
        "response_style": {
            "tone": "plain-language",
            "max_sentences": 2,
            "top_response_tokens": [token for token, _count in top_response_tokens],
        },
    }


def _resolve_output_dir(output_root: str | Path | None, training_run_id: str) -> Path:
    root = Path(output_root or ".local/modelbook-artifacts")
    return (root / _safe_name(training_run_id)).resolve()


def _require_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TinyChatPackagingError(f"missing required modelbook field {key!r}")
    return value.strip()


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.lower())


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip(".-")
    return cleaned or "artifact"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _write_gguf_fixture(path: Path, payload: Mapping[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True).encode("utf-8")
    path.write_bytes(b"GGUF" + (3).to_bytes(4, "little") + len(body).to_bytes(8, "little") + body)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_file(path: Path, *, role: str, media_type: str) -> TinyChatArtifactFile:
    return TinyChatArtifactFile(
        logical_name=path.name,
        role=role,
        media_type=media_type,
        path=path,
        size_bytes=path.stat().st_size,
        sha256_hex=_sha256_file(path),
    )


def _file_ref(path: Path) -> str:
    return path.resolve().as_uri()
