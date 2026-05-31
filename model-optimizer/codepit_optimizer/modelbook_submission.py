"""Canonical submission bridge for Modelbook Tiny Chat artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from .protocol import CodePitClient
from .tiny_chat_packager import TinyChatArtifactPackage


class ModelbookSubmissionError(RuntimeError):
    """Raised when a Modelbook artifact package cannot enter the V2 submission path."""


@dataclass(frozen=True)
class ModelbookSubmissionResult:
    challenge_id: str
    submission_id: str
    state: str
    client_submission_id: str


def submit_tiny_chat_package(
    *,
    client: CodePitClient,
    agent_id: str,
    training_run_id: str,
    modelbook: Mapping[str, Any],
    context: Mapping[str, Any],
    package: TinyChatArtifactPackage,
    recipe_kind: str,
    quantization_profile: str,
    challenge_id: str | None = None,
    client_submission_id: str | None = None,
) -> ModelbookSubmissionResult:
    """Create, upload, and attach a canonical Submission for a Tiny Chat run."""

    challenge = _resolve_challenge(client, challenge_id, context)
    resolved_challenge_id = _require_text(challenge, "challenge_id")
    manifest = build_tiny_chat_manifest(
        challenge=challenge,
        modelbook=modelbook,
        context=context,
        package=package,
        recipe_kind=recipe_kind,
        quantization_profile=quantization_profile,
    )
    resolved_client_submission_id = client_submission_id or _default_client_submission_id(
        agent_id=agent_id,
        challenge_id=resolved_challenge_id,
        training_run_id=training_run_id,
        manifest=manifest,
    )

    try:
        created = client.create_submission(
            {
                "protocol_version": "v1",
                "client_submission_id": resolved_client_submission_id,
                "agent_id": agent_id,
                "challenge_id": resolved_challenge_id,
                "manifest_schema_version": "0.2.0",
                "manifest_envelope": manifest,
            }
        )
    except Exception as error:
        raise ModelbookSubmissionError(
            f"canonical submission create failed: {error}",
        ) from error

    _drive_uploads(client, created, package)

    submission_id = _require_text(created, "submission_id")
    try:
        client.submit_training_run(training_run_id, {"submission_id": submission_id})
    except Exception as error:
        raise ModelbookSubmissionError(
            f"TrainingRun submission attach failed: {error}",
        ) from error
    return ModelbookSubmissionResult(
        challenge_id=resolved_challenge_id,
        submission_id=submission_id,
        state=str(created.get("state") or ""),
        client_submission_id=resolved_client_submission_id,
    )


def build_tiny_chat_manifest(
    *,
    challenge: Mapping[str, Any],
    modelbook: Mapping[str, Any],
    context: Mapping[str, Any],
    package: TinyChatArtifactPackage,
    recipe_kind: str,
    quantization_profile: str,
) -> dict[str, Any]:
    """Build the manifest envelope admitted by the engine for GGUF/Ollama artifacts."""

    context_modelbook = context.get("modelbook") if isinstance(context.get("modelbook"), Mapping) else {}
    source_model = str(
        modelbook.get("base_model_ref")
        or context_modelbook.get("base_model_ref")
        or package.provenance.get("base_model_ref")
        or "hf://Qwen/Qwen2.5-0.5B-Instruct"
    )
    model_class = str(
        modelbook.get("model_class")
        or context_modelbook.get("model_class")
        or "chat-causal-small"
    )
    artifact_lane = str(
        modelbook.get("artifact_lane")
        or context_modelbook.get("artifact_lane")
        or "ollama-gguf-local"
    )
    benchmark_target_version = _require_text(challenge, "benchmark_target_version")

    if challenge.get("artifact_lane") != artifact_lane:
        raise ModelbookSubmissionError(
            "challenge artifact_lane does not match Modelbook artifact lane: "
            f"{challenge.get('artifact_lane')} != {artifact_lane}"
        )
    admission_rules = list(challenge.get("model_class_admission_rules") or [])
    if admission_rules and model_class not in admission_rules:
        raise ModelbookSubmissionError(
            f"challenge does not admit Modelbook model_class {model_class!r}"
        )

    primary_files = [file for file in package.files if file.role == "primary_model"]
    if len(primary_files) != 1:
        raise ModelbookSubmissionError("Tiny Chat package must contain exactly one primary_model")
    primary = primary_files[0]
    supporting_files = [file.logical_name for file in package.files if file.logical_name != primary.logical_name]

    return {
        "schema_version": "0.2.0",
        "artifact_lane": artifact_lane,
        "model_class": model_class,
        "benchmark_target_ref": benchmark_target_version,
        "source_model": {
            "identifier": source_model,
            "family": model_class,
        },
        "runtime_target": {
            "environment_family": "local-ollama",
            "runtime": "ollama",
            "constraints": {
                "quantization_profile": quantization_profile,
                "prompt_template": "{{ .Prompt }}\\n{{ .Response }}",
                "modelfile_logical_name": "Modelfile",
                "provenance_logical_name": "provenance.json",
            },
        },
        "optimization": {
            "methods": [recipe_kind, quantization_profile],
            "notes": "Deterministic Tiny Chat Modelbook training package for local Ollama.",
        },
        "files": [
            {
                "logical_name": file.logical_name,
                "role": file.role,
                "media_type": file.media_type,
                "size_bytes": file.size_bytes,
                "sha256": file.sha256_hex,
            }
            for file in package.files
        ],
        "entrypoint": {
            "primary_model_logical_name": primary.logical_name,
            "supporting_files": supporting_files,
        },
        "claimed_metrics": {
            "training_example_count": float(package.provenance.get("training_example_count") or 0),
        },
    }


def _resolve_challenge(
    client: CodePitClient,
    challenge_id: str | None,
    context: Mapping[str, Any],
) -> Mapping[str, Any]:
    if challenge_id:
        return client.read_challenge(challenge_id)

    constraints = context.get("verifier_constraints")
    if isinstance(constraints, Mapping) and isinstance(constraints.get("challenge_id"), str):
        return client.read_challenge(str(constraints["challenge_id"]))

    next_response = client.next_challenge()
    challenge = next_response.get("challenge")
    if not isinstance(challenge, Mapping):
        raise ModelbookSubmissionError("no eligible benchmark challenge available for Modelbook submission")
    resolved_id = _require_text(challenge, "challenge_id")
    return client.read_challenge(resolved_id)


def _drive_uploads(
    client: CodePitClient,
    create_response: Mapping[str, Any],
    package: TinyChatArtifactPackage,
) -> None:
    orchestration = create_response.get("upload_orchestration")
    if not isinstance(orchestration, Mapping) or orchestration.get("kind") != "presigned-urls":
        raise ModelbookSubmissionError("submission response did not include presigned upload URLs")
    instructions = orchestration.get("files")
    if not isinstance(instructions, list) or not instructions:
        raise ModelbookSubmissionError("submission response did not include upload instructions")

    by_logical_name = {file.logical_name: file for file in package.files}
    for instruction in instructions:
        if not isinstance(instruction, Mapping):
            raise ModelbookSubmissionError("upload instruction is not an object")
        logical_name = str(instruction.get("logical_name") or "")
        upload_url = str(instruction.get("upload_url") or "")
        media_type = str(instruction.get("media_type") or "")
        expected_size = instruction.get("size_bytes")
        expected_sha256 = str(instruction.get("sha256") or "")
        file = by_logical_name.get(logical_name)
        if file is None:
            raise ModelbookSubmissionError(f"Tiny Chat package is missing upload file {logical_name!r}")
        if not upload_url or not media_type:
            raise ModelbookSubmissionError(f"upload instruction is incomplete for {logical_name!r}")
        if media_type != file.media_type:
            raise ModelbookSubmissionError(f"upload media_type mismatch for {logical_name!r}")
        if isinstance(expected_size, int) and expected_size != file.size_bytes:
            raise ModelbookSubmissionError(f"upload size mismatch for {logical_name!r}")
        if expected_sha256 and expected_sha256.lower() != file.sha256_hex.lower():
            raise ModelbookSubmissionError(f"upload sha256 mismatch for {logical_name!r}")
        if instruction.get("already_uploaded") is True:
            continue
        try:
            client.put_bytes(upload_url, file.path.read_bytes(), content_type=media_type)
        except Exception as error:
            raise ModelbookSubmissionError(f"upload failed for {logical_name!r}: {error}") from error


def _default_client_submission_id(
    *,
    agent_id: str,
    challenge_id: str,
    training_run_id: str,
    manifest: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    raw = f"modelbook-{agent_id}-{challenge_id}-{training_run_id}-{digest}"
    return raw[:128]


def _require_text(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ModelbookSubmissionError(f"missing required field {key!r}")
    return value.strip()
