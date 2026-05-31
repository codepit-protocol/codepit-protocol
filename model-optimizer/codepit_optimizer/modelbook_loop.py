"""Modelbook-driven agent iteration loop.

The historical ``orchestrator.run_optimizer_agent_forever`` flow polls the V1
challenge endpoint, runs ONNX recipes (graph-optimization / dynamic-int8) on a
small encoder, and submits a manifest envelope to the verifier.

This module is the parallel path for the V2 SML Modelbook workspace. An
agent attached here pulls an open Modelbook (Qwen2.5-0.5B-Instruct, base
"chat-causal-small", artifact lane "ollama-gguf-local"), records a
training run, emits decisions + events, performs a deterministic Tiny Chat
training fixture, and registers checksum-backed artifact files.

The strategic decisions (recipe pick, hyperparameters, export choice) can be
either:
- **Brain-driven (default when a brain is wired)** - each decision is a real
  LLM call to the engine's ``/v2/brain/generate`` endpoint.
  ``brain_provider`` + ``brain_model`` on each decision row reflect the
  upstream LLM the engine routed to.
- **Heuristic (fallback)** - when no brain is configured the iteration uses
  deterministic helpers and tags decisions with ``brain_provider="heuristic"``.
  Useful for tests and dev without network.

The dashboard reads ``brain_provider`` / ``brain_model`` per decision so an
operator can always tell which mode produced a given run's choices.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from .brain_providers.managed import ManagedBrainResponse
from .modelbook_submission import ModelbookSubmissionError, submit_tiny_chat_package
from .protocol import CodePitClient
from .gguf_build_pipeline import make_env_gguf_builder
from .tiny_chat_packager import TinyChatPackagingError, train_and_package_tiny_chat


class ModelbookLoopError(RuntimeError):
    """Raised when the Modelbook iteration cannot make forward progress."""


class BrainCallFailed(ModelbookLoopError):
    """Raised when a brain LLM call fails (network, schema parse, validation).

    Iteration aborts on first failure rather than degrading to a stub —
    callers who opted into LLM-driven decisions shouldn't silently fall
    back to deterministic heuristics. Use ``brain=None`` in the config to
    opt out instead.
    """


class BrainLike(Protocol):
    """Minimal contract the iteration needs from a brain provider.

    ``ManagedBrainProvider`` from :mod:`brain_providers.managed` implements
    this. Tests can pass a fake that returns canned ``ManagedBrainResponse``
    values without any HTTP.
    """

    def generate_with_metadata(
        self,
        *,
        prompt: str,
        action_id: str,
        attempt: int,
        tier: str,
        schema: Mapping[str, Any] | None = None,
        system: str | None = None,
    ) -> ManagedBrainResponse: ...


@dataclass
class ModelbookIterationConfig:
    """Inputs for a single ``run_modelbook_iteration`` pass."""

    #: Pin to a specific Modelbook id. ``None`` → first available.
    modelbook_id: str | None = None
    #: Pin to a specific recipe. Must be in ``policy.allowed_training_methods``.
    #: ``None`` → brain (or heuristic) chooses.
    recipe_kind: str | None = None
    #: Directory where Tiny Chat artifacts are written. ``None`` uses
    #: ``.local/modelbook-artifacts`` under the current process directory.
    artifact_output_dir: str | Path | None = None
    #: When set, every strategic decision is made by an LLM call through
    #: this provider. When ``None``, the iteration falls back to
    #: deterministic heuristics and tags decisions with ``"heuristic"``.
    brain: BrainLike | None = None
    #: Tier passed to brain calls. Resolved on the engine to a real
    #: provider+model pair. Default ``"cheap"`` matches the dashboard.
    brain_tier: str = "cheap"
    #: When true, the local Tiny Chat package is sent through the canonical
    #: /v1/submissions upload path, then attached back to the TrainingRun.
    submit: bool = False
    #: Optional verifier challenge id. If absent, the loop asks
    #: /v1/challenges/next after packaging.
    challenge_id: str | None = None
    #: Optional retry key for the canonical submission create call.
    client_submission_id: str | None = None
    #: When true, ask the brain for one best-effort first-class social post/reply
    #: and send it through ``POST /v2/modelbooks/:id/posts``. Failures are
    #: recorded on the result and never block training, packaging, or submit.
    social_posts_enabled: bool = False


@dataclass
class ModelbookIterationResult:
    modelbook_id: str | None
    training_run_id: str | None
    recipe_kind: str | None
    decisions_recorded: int = 0
    events_emitted: int = 0
    artifact_set_id: str | None = None
    submission_id: str | None = None
    submission_state: str | None = None
    challenge_id: str | None = None
    skipped_reason: str | None = None
    stub_training_used: bool = False
    brain_driven: bool = False
    brain_provider: str | None = None
    brain_model: str | None = None
    social_posts_created: int = 0
    social_post_failures: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class _DecisionMaterials:
    """Bundle the iteration assembles before recording a decision row."""

    summary: str
    rationale: str
    selected_inputs: dict[str, Any]
    rejected_options: list[Any] = field(default_factory=list)
    risk_notes: list[str] = field(default_factory=list)
    brain_provider: str | None = None
    brain_model: str | None = None


@dataclass
class _SocialAction:
    action: str
    title: str | None = None
    body: str | None = None
    parent_post_id: str | None = None


def run_modelbook_iteration(
    client: CodePitClient,
    config: ModelbookIterationConfig | None = None,
) -> ModelbookIterationResult:
    """Run one Modelbook training pass against the engine.

    Returns immediately with ``skipped_reason`` populated if no Modelbook is
    available, so the supervisor loop can back off without raising.
    """

    cfg = config or ModelbookIterationConfig()

    modelbook = _pick_modelbook(client, pin_id=cfg.modelbook_id)
    if modelbook is None:
        return ModelbookIterationResult(
            modelbook_id=None,
            training_run_id=None,
            recipe_kind=None,
            skipped_reason="no_available_modelbook",
            stub_training_used=False,
        )

    modelbook_id = str(modelbook["modelbook_id"])
    context = client.read_modelbook_context(modelbook_id)
    policy = context.get("policy") or {}
    allowed_methods = list(policy.get("allowed_training_methods") or [])
    allowed_exports = list(policy.get("allowed_export_targets") or [])

    if not allowed_methods:
        raise ModelbookLoopError(
            "policy.allowed_training_methods is empty — Modelbook is not workable",
        )
    if not allowed_exports:
        raise ModelbookLoopError(
            "policy.allowed_export_targets is empty — cannot register an artifact",
        )

    # ── Decision 1: recipe ────────────────────────────────────────────
    if cfg.recipe_kind is not None:
        if cfg.recipe_kind not in allowed_methods:
            raise ModelbookLoopError(
                f"requested recipe_kind {cfg.recipe_kind!r} is not in "
                f"allowed methods {allowed_methods}",
            )
        recipe_decision = _heuristic_recipe(cfg.recipe_kind, allowed_methods)
    elif cfg.brain is not None:
        recipe_decision = _brain_pick_recipe(
            cfg.brain, modelbook, allowed_methods, cfg.brain_tier
        )
    else:
        recipe_decision = _heuristic_recipe(allowed_methods[0], allowed_methods)
    recipe_kind = str(recipe_decision.selected_inputs["recipe_kind"])

    # ── Decision 2: hyperparameters ──────────────────────────────────
    if cfg.brain is not None:
        hyperparam_decision = _brain_pick_hyperparameters(
            cfg.brain, modelbook, recipe_kind, policy, cfg.brain_tier
        )
    else:
        hyperparam_decision = _heuristic_hyperparameters(recipe_kind)
    hyperparams = dict(hyperparam_decision.selected_inputs)

    # ── Decision 3: export quantization ──────────────────────────────
    if cfg.brain is not None:
        export_decision = _brain_pick_export(
            cfg.brain, modelbook, allowed_exports, cfg.brain_tier
        )
    else:
        export_decision = _heuristic_export(allowed_exports)
    quantization_profile = str(export_decision.selected_inputs["quantization_profile"])
    if quantization_profile not in allowed_exports:
        raise ModelbookLoopError(
            f"brain proposed export {quantization_profile!r} not in "
            f"policy.allowed_export_targets {allowed_exports}",
        )

    # ── Create the run + emit decisions/events in order ──────────────
    run_response = client.create_training_run(
        modelbook_id,
        objective=(
            "Specialize the tiny chat model toward concise, helpful replies "
            "while staying within policy caps."
        ),
        recipe_kind=recipe_kind,
    )
    run = run_response.get("run") or {}
    training_run_id = str(run.get("training_run_id") or run.get("id") or "")
    if not training_run_id:
        raise ModelbookLoopError(
            f"engine did not return a training_run_id for modelbook {modelbook_id}",
        )

    result = ModelbookIterationResult(
        modelbook_id=modelbook_id,
        training_run_id=training_run_id,
        recipe_kind=recipe_kind,
        events_emitted=1,  # the engine emits "run.started" on create
        brain_driven=cfg.brain is not None,
        brain_provider=recipe_decision.brain_provider,
        brain_model=recipe_decision.brain_model,
        notes=[
            f"modelbook base_model_ref={modelbook.get('base_model_ref')}",
            f"policy allowed_training_methods={allowed_methods}",
        ],
    )

    recipe_decision_id = _record_decision(
        client, training_run_id, "recipe", recipe_decision
    )
    result.decisions_recorded += 1

    _record_decision(client, training_run_id, "hyperparameters", hyperparam_decision)
    result.decisions_recorded += 1

    export_decision_id = _record_decision(
        client, training_run_id, "export", export_decision
    )
    result.decisions_recorded += 1

    _post_feed_update(
        client,
        training_run_id,
        title="Training plan chosen",
        message=(
            f"I chose {recipe_kind} and {quantization_profile} packaging for "
            "this Tiny Chat run. The goal is a small local model that gives "
            "more helpful replies while staying inside the Modelbook rules."
        ),
        metadata={
            "phase": "planning",
            "decision_id": recipe_decision_id or export_decision_id,
        },
    )
    result.events_emitted += 1

    _run_social_step(
        client=client,
        config=cfg,
        result=result,
        modelbook_id=modelbook_id,
        training_run_id=training_run_id,
        modelbook=modelbook,
        context=context,
        recipe_kind=recipe_kind,
        quantization_profile=quantization_profile,
        decisions=[recipe_decision, hyperparam_decision, export_decision],
    )

    client.create_run_event(
        training_run_id,
        {
            "event_type": "training.started",
            "message": (
                "Agent began Tiny Chat training with brain-driven decisions."
                if cfg.brain is not None
                else "Agent began Tiny Chat training with heuristic decisions."
            ),
            "metadata": {
                "brain_driven_decisions": cfg.brain is not None,
                "recipe_kind": recipe_kind,
                "quantization_profile": quantization_profile,
            },
        },
    )
    result.events_emitted += 1

    try:
        package = train_and_package_tiny_chat(
            modelbook=modelbook,
            context=context,
            training_run_id=training_run_id,
            recipe_kind=recipe_kind,
            hyperparameters=hyperparams,
            quantization_profile=quantization_profile,
            output_root=cfg.artifact_output_dir,
            gguf_build=make_env_gguf_builder(),
        )
    except TinyChatPackagingError as error:
        client.create_run_event(
            training_run_id,
            {
                "event_type": "training.failed",
                "message": "Tiny Chat training or packaging failed before artifact registration.",
                "metadata": {
                    "error_code": "tiny_chat_packaging_failed",
                    "error_message": str(error),
                    "recipe_kind": recipe_kind,
                },
            },
        )
        result.events_emitted += 1
        raise ModelbookLoopError(
            f"tiny-chat training/package failed for run {training_run_id}: {error}",
        ) from error

    for event in package.progress_events:
        client.create_run_event(training_run_id, event)
        result.events_emitted += 1

    client.create_run_event(
        training_run_id,
        {
            "event_type": "training.complete",
            "message": "Tiny Chat training and packaging completed. Artifact registration is next.",
            "metadata": {
                "artifact_dir": str(package.output_dir),
                "quantization_profile": quantization_profile,
                "checksum_ref": package.checksum_ref,
            },
        },
    )
    result.events_emitted += 1

    artifact_response = client.create_artifact_set(
        training_run_id,
        {
            "artifact_lane": modelbook.get("artifact_lane", "ollama-gguf-local"),
            "primary_artifact_ref": package.primary_artifact_ref,
            "adapter_ref": package.adapter_ref,
            "merged_model_ref": package.merged_model_ref,
            "gguf_ref": package.gguf_ref,
            "modelfile_ref": package.modelfile_ref,
            "checksum_ref": package.checksum_ref,
            "quantization_profile": quantization_profile,
            "dataset_shard_ids": package.dataset_shard_ids,
            "provenance": {
                **package.provenance,
                "brain_driven_decisions": cfg.brain is not None,
                "brain_provider": result.brain_provider,
                "brain_model": result.brain_model,
            },
        },
    )
    artifact = artifact_response.get("artifact_set") or artifact_response
    result.artifact_set_id = (
        str(artifact.get("artifact_set_id"))
        if isinstance(artifact, Mapping) and artifact.get("artifact_set_id")
        else None
    )
    result.notes.append(f"artifact_output_dir={package.output_dir}")

    _post_feed_update(
        client,
        training_run_id,
        title="Model package ready",
        message=(
            "I finished the Tiny Chat package and recorded checksums so CodePit "
            "can verify the exact files I submit."
        ),
        metadata={
            "phase": "packaging",
            "artifact_set_id": result.artifact_set_id,
        },
    )
    result.events_emitted += 1

    if cfg.submit:
        agent_id = _resolve_agent_id(client, context)
        try:
            submission = submit_tiny_chat_package(
                client=client,
                agent_id=agent_id,
                training_run_id=training_run_id,
                modelbook=modelbook,
                context=context,
                package=package,
                recipe_kind=recipe_kind,
                quantization_profile=quantization_profile,
                challenge_id=cfg.challenge_id,
                client_submission_id=cfg.client_submission_id,
            )
        except ModelbookSubmissionError as error:
            client.create_run_event(
                training_run_id,
                {
                    "event_type": "submission.failed",
                    "message": "Agent could not submit the Tiny Chat artifact package to the verifier.",
                    "metadata": {
                        "error_code": "modelbook_submission_failed",
                        "error_message": str(error),
                    },
                },
            )
            result.events_emitted += 1
            raise ModelbookLoopError(
                f"modelbook submission failed for run {training_run_id}: {error}",
            ) from error
        result.submission_id = submission.submission_id
        result.submission_state = submission.state
        result.challenge_id = submission.challenge_id
        result.events_emitted += 1  # submit route records verifier.submitted/terminal state
        result.notes.append(f"submission_id={submission.submission_id}")
        _post_feed_update(
            client,
            training_run_id,
            title="Submitted for benchmark",
            message=(
                "I sent the package to CodePit's verifier. The official result "
                "will come from the platform benchmark, not from my own report."
            ),
            metadata={
                "phase": "submitted",
                "submission_id": submission.submission_id,
            },
        )
        result.events_emitted += 1

    return result


def run_modelbook_loop(
    client: CodePitClient,
    config: ModelbookIterationConfig | None = None,
    *,
    max_iterations: int | None = None,
    idle_sleep_seconds: float = 5.0,
    sleep: Any = time.sleep,
) -> list[ModelbookIterationResult]:
    """Repeatedly call ``run_modelbook_iteration``."""

    cfg = config or ModelbookIterationConfig()
    results: list[ModelbookIterationResult] = []
    completed_iterations = 0
    while True:
        result = run_modelbook_iteration(client, cfg)
        results.append(result)
        if result.skipped_reason is None:
            completed_iterations += 1
            if max_iterations is not None and completed_iterations >= max_iterations:
                return results
        else:
            sleep(idle_sleep_seconds)


# --------------------------------------------------------------------------
# Engine wiring helpers
# --------------------------------------------------------------------------


def _record_decision(
    client: CodePitClient,
    training_run_id: str,
    decision_type: str,
    decision: _DecisionMaterials,
) -> str | None:
    response = client.create_run_decision(
        training_run_id,
        {
            "decision_type": decision_type,
            "summary": decision.summary,
            "rationale": decision.rationale,
            "selected_inputs": decision.selected_inputs,
            "rejected_options": decision.rejected_options,
            "risk_notes": decision.risk_notes,
            "brain_provider": decision.brain_provider or "heuristic",
            "brain_model": decision.brain_model or "modelbook-loop-heuristic",
        },
    )
    payload = response.get("decision") if isinstance(response, Mapping) else None
    if not isinstance(payload, Mapping):
        return None
    decision_id = (
        payload.get("agent_decision_id")
        or payload.get("decision_id")
        or payload.get("id")
    )
    return str(decision_id) if decision_id else None


def _post_feed_update(
    client: CodePitClient,
    training_run_id: str,
    *,
    title: str,
    message: str,
    metadata: Mapping[str, Any],
) -> None:
    safe_metadata = {
        key: value
        for key, value in metadata.items()
        if value is not None
    }
    client.create_run_event(
        training_run_id,
        {
            "event_type": "feed.agent_post",
            "message": message,
            "metadata": {
                "title": title,
                **safe_metadata,
            },
        },
    )


_SOCIAL_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["action"],
    "properties": {
        "action": {"type": "string", "enum": ["post", "reply", "silent"]},
        "title": {"type": "string"},
        "body": {"type": "string"},
        "parent_post_id": {"type": "string"},
    },
    "additionalProperties": False,
}


def _run_social_step(
    *,
    client: CodePitClient,
    config: ModelbookIterationConfig,
    result: ModelbookIterationResult,
    modelbook_id: str,
    training_run_id: str,
    modelbook: Mapping[str, Any],
    context: Mapping[str, Any],
    recipe_kind: str,
    quantization_profile: str,
    decisions: list[_DecisionMaterials],
) -> None:
    if not config.social_posts_enabled:
        return
    if config.brain is None:
        result.notes.append("social_post_skipped=no_brain")
        return

    try:
        social = _brain_pick_social_action(
            config.brain,
            modelbook=modelbook,
            context=context,
            recipe_kind=recipe_kind,
            quantization_profile=quantization_profile,
            decisions=decisions,
            tier=config.brain_tier,
        )
        if social.action == "silent":
            result.notes.append("social_post_skipped=silent")
            return

        if not social.body:
            raise BrainCallFailed("brain social response missing body for post/reply")
        payload: dict[str, Any] = {
            "training_run_id": training_run_id,
            "client_post_id": f"{training_run_id}:social:1",
            "body": social.body[:800],
        }
        if social.title:
            payload["title"] = social.title[:140]
        if social.action == "reply":
            if not social.parent_post_id:
                raise BrainCallFailed("brain social reply missing parent_post_id")
            payload["parent_post_id"] = social.parent_post_id

        client.create_modelbook_post(modelbook_id, payload)
        result.social_posts_created += 1
    except Exception as error:
        result.social_post_failures += 1
        preview = str(error).replace("\n", " ")[:180]
        result.notes.append(f"social_post_failed={type(error).__name__}:{preview}")


def _brain_pick_social_action(
    brain: BrainLike,
    *,
    modelbook: Mapping[str, Any],
    context: Mapping[str, Any],
    recipe_kind: str,
    quantization_profile: str,
    decisions: list[_DecisionMaterials],
    tier: str,
) -> _SocialAction:
    system = (
        "You are an autonomous CodePit training agent writing a short public "
        "feed update. The post is narrative only: no official ranking, proof, "
        "reward, or payout claims. Respond ONLY with valid JSON."
    )
    decision_summaries = [
        {
            "summary": decision.summary,
            "rationale": decision.rationale,
            "selected_inputs": decision.selected_inputs,
        }
        for decision in decisions
    ]
    prompt = (
        f"Modelbook: {modelbook.get('display_name') or modelbook.get('modelbook_id')}\n"
        f"Base model: {modelbook.get('base_model_ref')}\n"
        f"Recipe: {recipe_kind}\n"
        f"Quantization: {quantization_profile}\n"
        f"Policy: {json.dumps(context.get('policy') or {}, default=str)[:1200]}\n"
        f"Decisions: {json.dumps(decision_summaries, default=str)[:1800]}\n\n"
        "Choose one action:\n"
        "- post: publish a short top-level update explaining what you tried and why a small model matters.\n"
        "- reply: reply to a known parent_post_id only if the context gives you one.\n"
        "- silent: do not publish if there is no useful public update.\n"
        "Keep body under 800 characters. Do not claim official verification."
    )
    response = _call_brain(
        brain,
        action_id="modelbook-social",
        attempt=1,
        tier=tier,
        prompt=prompt + "\n\nReturn JSON only:\n" + json.dumps(_SOCIAL_SCHEMA),
        schema=None,
        system=system,
    )
    parsed = _parse_brain_json(response.content, decision="social")
    action = _require_str(parsed, "action", decision="social").lower()
    if action not in {"post", "reply", "silent"}:
        raise BrainCallFailed(
            f"brain social response action must be post, reply, or silent; got {action!r}"
        )
    if action == "silent":
        return _SocialAction(action="silent")

    body = _require_str(parsed, "body", decision="social")
    title_raw = parsed.get("title")
    parent_raw = parsed.get("parent_post_id")
    return _SocialAction(
        action=action,
        title=title_raw.strip() if isinstance(title_raw, str) and title_raw.strip() else None,
        body=body,
        parent_post_id=parent_raw.strip()
        if isinstance(parent_raw, str) and parent_raw.strip()
        else None,
    )


def _pick_modelbook(
    client: CodePitClient,
    *,
    pin_id: str | None,
) -> Mapping[str, Any] | None:
    response = client.list_available_modelbooks()
    items = response.get("items") or []
    if not items:
        return None
    if pin_id is None:
        return items[0]
    for item in items:
        if str(item.get("modelbook_id")) == pin_id:
            return item
    raise ModelbookLoopError(
        f"pinned modelbook_id {pin_id} not present in /v2/modelbooks/available",
    )


def _resolve_agent_id(client: CodePitClient, context: Mapping[str, Any]) -> str:
    if client.agent_id:
        return client.agent_id
    assigned_agent = context.get("assigned_agent")
    if isinstance(assigned_agent, Mapping):
        agent_id = assigned_agent.get("agent_id")
        if isinstance(agent_id, str) and agent_id.strip():
            return agent_id.strip()
    raise ModelbookLoopError(
        "cannot submit Modelbook package without an agent id in the client or context",
    )


# --------------------------------------------------------------------------
# Heuristic decisions (no LLM)
# --------------------------------------------------------------------------


def _heuristic_recipe(recipe_kind: str, allowed: list[str]) -> _DecisionMaterials:
    if recipe_kind == "lora":
        rationale = (
            "LoRA is the first allowed method and the lowest-risk way to bias "
            "the base model toward the Modelbook objective."
        )
    elif recipe_kind == "qlora":
        rationale = (
            "QLoRA reduces VRAM at training time while keeping the same "
            "adapter shape. Selected from allowed methods: " + ", ".join(allowed)
        )
    else:
        rationale = (
            f"Selected {recipe_kind} from policy.allowed_training_methods "
            f"({', '.join(allowed)})."
        )
    return _DecisionMaterials(
        summary=f"Selected recipe {recipe_kind}",
        rationale=rationale,
        selected_inputs={"recipe_kind": recipe_kind},
        rejected_options=[m for m in allowed if m != recipe_kind],
        risk_notes=["Heuristic pick — no LLM was consulted for this decision."],
    )


def _heuristic_hyperparameters(recipe_kind: str) -> _DecisionMaterials:
    base: dict[str, Any] = {
        "rank": 8,
        "alpha": 16,
        "dropout": 0.05,
        "learning_rate": 1e-4,
        "epochs": 1,
    }
    if recipe_kind == "qlora":
        base["bits"] = 4
    return _DecisionMaterials(
        summary="LoRA hyperparameters within policy caps",
        rationale=(
            "Heuristic defaults that mirror what a baseline adapter run would target."
        ),
        selected_inputs=base,
        risk_notes=["Heuristic pick — no LLM was consulted for this decision."],
    )


def _heuristic_export(allowed_exports: list[str]) -> _DecisionMaterials:
    quantization = allowed_exports[0]
    return _DecisionMaterials(
        summary=f"Export quantization {quantization}",
        rationale=(
            "First allowed quantization profile for the local Ollama package."
        ),
        selected_inputs={"quantization_profile": quantization},
        rejected_options=[q for q in allowed_exports if q != quantization],
        risk_notes=["Heuristic pick — no LLM was consulted for this decision."],
    )


# --------------------------------------------------------------------------
# Brain-driven decisions (real LLM via /v2/brain/generate)
# --------------------------------------------------------------------------


_RECIPE_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["recipe_kind", "rationale"],
    "properties": {
        "recipe_kind": {"type": "string"},
        "rationale": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
    },
    "additionalProperties": False,
}

_HP_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["learning_rate", "epochs", "rationale"],
    "properties": {
        "learning_rate": {"type": "number"},
        "epochs": {"type": "integer"},
        "rank": {"type": "integer"},
        "alpha": {"type": "integer"},
        "dropout": {"type": "number"},
        "bits": {"type": "integer"},
        "rationale": {"type": "string"},
    },
    "additionalProperties": False,
}

_EXPORT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "required": ["quantization_profile", "rationale"],
    "properties": {
        "quantization_profile": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "additionalProperties": False,
}


def _brain_pick_recipe(
    brain: BrainLike,
    modelbook: Mapping[str, Any],
    allowed: list[str],
    tier: str,
) -> _DecisionMaterials:
    system = (
        "You are an autonomous AI training agent on the CodePit V2 network. "
        "You pick fine-tuning recipes for small open-weight models. "
        "Respond ONLY with valid JSON. No prose, no markdown fences. "
        "The recipe_kind field MUST be one of the exact strings in the "
        "Allowed list — do not invent new names, do not capitalize, do not "
        "abbreviate."
    )
    prompt = (
        f"Modelbook display_name: {modelbook.get('display_name')}\n"
        f"Base model: {modelbook.get('base_model_ref')}\n"
        f"Model class: {modelbook.get('model_class')}\n"
        f"Artifact lane: {modelbook.get('artifact_lane')}\n"
        f"Allowed training methods (pick EXACTLY one of these strings, "
        f"verbatim): {json.dumps(allowed)}\n\n"
        "Pick ONE training method from the Allowed list above. The value of "
        f"recipe_kind in your response MUST equal one of: {json.dumps(allowed)}. "
        "Then explain in 1–2 sentences why it is the right pick for this "
        "Modelbook. Optionally list short risk notes."
    )
    response = _call_brain(
        brain,
        action_id="modelbook-recipe",
        attempt=1,
        tier=tier,
        prompt=prompt + "\n\nReturn JSON only:\n" + json.dumps(_RECIPE_SCHEMA),
        schema=None,
        system=system,
    )
    parsed = _parse_brain_json(response.content, decision="recipe")
    recipe_kind = _pick_allowed_value(
        parsed,
        candidate_keys=("recipe_kind", "recipe", "method", "training_method"),
        allowed=allowed,
        decision="recipe",
        raw_content=response.content,
    )
    rationale = _require_str(parsed, "rationale", decision="recipe")
    risks = parsed.get("risks") or []
    if not isinstance(risks, list):
        risks = []
    return _DecisionMaterials(
        summary=f"Selected recipe {recipe_kind}",
        rationale=rationale,
        selected_inputs={"recipe_kind": recipe_kind},
        rejected_options=[m for m in allowed if m != recipe_kind],
        risk_notes=[str(r) for r in risks if isinstance(r, str)],
        brain_provider=response.provider,
        brain_model=response.model,
    )


def _brain_pick_hyperparameters(
    brain: BrainLike,
    modelbook: Mapping[str, Any],
    recipe_kind: str,
    policy: Mapping[str, Any],
    tier: str,
) -> _DecisionMaterials:
    system = (
        "You are an autonomous AI training agent on the CodePit V2 network. "
        "You pick hyperparameters for fine-tuning small open-weight models. "
        "Respond ONLY with valid JSON matching the provided schema."
    )
    prompt = (
        f"Recipe: {recipe_kind}\n"
        f"Base model: {modelbook.get('base_model_ref')}\n"
        f"Model class: {modelbook.get('model_class')}\n"
        f"Policy max_budget_codepit: {policy.get('max_budget_codepit')}\n"
        f"Policy requires_publish_approval: {policy.get('requires_publish_approval')}\n\n"
        "Suggest reasonable LoRA hyperparameters for this base model. "
        "Keep epochs ≤ 3 and learning_rate ≤ 1e-3. If the recipe is "
        "'qlora', include a 'bits' field (4 or 8). One-paragraph rationale."
    )
    response = _call_brain(
        brain,
        action_id="modelbook-hyperparams",
        attempt=1,
        tier=tier,
        prompt=prompt + "\n\nReturn JSON only:\n" + json.dumps(_HP_SCHEMA),
        schema=None,
        system=system,
    )
    parsed = _parse_brain_json(response.content, decision="hyperparameters")
    rationale = _require_str(parsed, "rationale", decision="hyperparameters")
    selected = {k: v for k, v in parsed.items() if k != "rationale"}
    return _DecisionMaterials(
        summary="LoRA hyperparameters within policy caps",
        rationale=rationale,
        selected_inputs=selected,
        brain_provider=response.provider,
        brain_model=response.model,
    )


def _brain_pick_export(
    brain: BrainLike,
    modelbook: Mapping[str, Any],
    allowed_exports: list[str],
    tier: str,
) -> _DecisionMaterials:
    system = (
        "You are an autonomous AI training agent on the CodePit V2 network. "
        "Respond ONLY with valid JSON matching the provided schema."
    )
    prompt = (
        f"Artifact lane: {modelbook.get('artifact_lane')}\n"
        f"Allowed export targets: {', '.join(allowed_exports)}\n\n"
        "Pick ONE quantization profile from the allowed list and explain "
        "in 1–2 sentences why."
    )
    response = _call_brain(
        brain,
        action_id="modelbook-export",
        attempt=1,
        tier=tier,
        prompt=prompt + "\n\nReturn JSON only:\n" + json.dumps(_EXPORT_SCHEMA),
        schema=None,
        system=system,
    )
    parsed = _parse_brain_json(response.content, decision="export")
    quantization = _pick_allowed_value(
        parsed,
        candidate_keys=("quantization_profile", "quantization", "profile", "export_target"),
        allowed=allowed_exports,
        decision="export",
        raw_content=response.content,
    )
    rationale = _require_str(parsed, "rationale", decision="export")
    return _DecisionMaterials(
        summary=f"Export quantization {quantization}",
        rationale=rationale,
        selected_inputs={"quantization_profile": quantization},
        rejected_options=[q for q in allowed_exports if q != quantization],
        brain_provider=response.provider,
        brain_model=response.model,
    )


def _call_brain(
    brain: BrainLike,
    *,
    action_id: str,
    attempt: int,
    tier: str,
    prompt: str,
    schema: Mapping[str, Any],
    system: str,
) -> ManagedBrainResponse:
    try:
        return brain.generate_with_metadata(
            prompt=prompt,
            action_id=action_id,
            attempt=attempt,
            tier=tier,
            schema=schema,
            system=system,
        )
    except Exception as error:  # pragma: no cover - thin re-raise
        raise BrainCallFailed(
            f"brain call '{action_id}' failed: {error}"
        ) from error


_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _parse_brain_json(content: str, *, decision: str) -> dict[str, Any]:
    cleaned = _JSON_FENCE_RE.sub("", content.strip()).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError as error:
        raise BrainCallFailed(
            f"brain {decision!r} response was not valid JSON: {error}; "
            f"first 200 chars: {cleaned[:200]!r}"
        ) from error
    if not isinstance(value, dict):
        raise BrainCallFailed(
            f"brain {decision!r} response root was {type(value).__name__}, expected object"
        )
    return value


def _require_str(payload: Mapping[str, Any], key: str, *, decision: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise BrainCallFailed(
            f"brain {decision!r} response missing required string field {key!r}"
        )
    return value.strip()


def _pick_allowed_value(
    payload: Mapping[str, Any],
    *,
    candidate_keys: tuple[str, ...],
    allowed: list[str],
    decision: str,
    raw_content: str,
) -> str:
    """Pull a string from the parsed LLM response, tolerating field-name drift.

    Tries each ``candidate_keys`` in order; falls back to scanning all
    string values in the payload for an exact match against ``allowed``.
    Raises ``BrainCallFailed`` with the raw LLM content in the message if
    no match is found, so operators can see exactly what the brain returned.
    """

    for key in candidate_keys:
        candidate = payload.get(key)
        if isinstance(candidate, str) and candidate.strip() in allowed:
            return candidate.strip()
    # Scan every string value in the payload.
    for value in payload.values():
        if isinstance(value, str) and value.strip() in allowed:
            return value.strip()
    # Last-ditch: scan the FULL raw content for any allowed substring.
    # Small models often name the right value in prose ("QLora is the right
    # option") even while the JSON field carries something hallucinated.
    haystack = raw_content.lower()
    for value in allowed:
        if value.lower() in haystack:
            return value
    preview = raw_content[:300] + ("…" if len(raw_content) > 300 else "")
    raise BrainCallFailed(
        f"brain {decision!r} response did not match any allowed value in "
        f"{allowed}. Tried keys {candidate_keys}. Raw content: {preview!r}"
    )
