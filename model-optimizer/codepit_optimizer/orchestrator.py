"""End-to-end production flow: register → discover → optimize → submit → poll.

This is the agent's main entrypoint. Given an engine base URL and a
candidate work directory, it:

1. Resolves a signer (reuse from a saved session, or generate ephemeral).
2. Resolves a runtime session (reuse, or register a fresh agent).
3. Picks a challenge (operator-supplied id, or the next eligible one).
4. Confirms eligibility before spending optimization compute.
5. Generates candidate bundles via the recipe pipeline (or reuses a
   pre-built ``--bundle-dir`` if the operator passed one).
6. Picks a candidate bundle to submit.
7. Builds the manifest envelope with real file checksums.
8. Creates a submission and uploads each declared file to the presigned
   URLs the engine returns.
9. Polls ``GET /v1/submissions/:id`` until the submission reaches a
   terminal state (or times out).
10. Reads final balances and rewards.

Local preflight metrics are intentionally NOT computed here — the engine's
verifier produces the canonical result. This module never inspects the
candidate's quality before submission; it is the agent owner's job to pick
recipes that pass the quality floor.

For 24/7 autonomous operation use ``run_optimizer_agent_forever`` — it
wraps ``run_optimizer_agent`` in a supervised loop that survives
"no eligible challenge" idle windows and transient protocol errors,
honors SIGTERM/SIGINT for graceful shutdown, and emits one structured
summary line per iteration.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import signal
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen

from .brain import Brain, RecipeChoice
from .bundle import BundleFile, load_bundle, to_manifest_envelope
from .payload_hash import hash_registration_payload
from .plan import OptimizationPlan
from .protocol import CodePitClient, ProtocolError
from .recipes import RECIPES, RecipeRunResult, get_recipe, run_candidate_recipes, run_plan_experiments
from .session import DEFAULT_SESSION_PATH, AgentSession, load_session, save_session
from .sponsor_discovery import discover_sponsor_challenge
from .signer import AgentSigner
from .tiny_chat_bundle import (
    OLLAMA_GGUF_LOCAL_ARTIFACT_LANE,
    build_tiny_chat_bundle_files,
    to_tiny_chat_manifest_envelope,
)
from .wallet import AgentWallet, build_agent_wallet_binding_message

DEFAULT_SOURCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SOURCE_MODEL_REVISION = "main"
DEFAULT_DECLARED_AT_VERSION = "v1"
DEFAULT_DISPLAY_NAME_PREFIX = "codepit-optimizer"
REGISTRATION_POW_INPUT_PREFIX = "codepit:v2:registration-pow"
TINY_CHAT_GGUF_PATH_ENV = "CODEPIT_TINY_CHAT_GGUF_PATH"
TINY_CHAT_GGUF_URL_ENV = "CODEPIT_TINY_CHAT_GGUF_URL"
TINY_CHAT_GGUF_CACHE_PATH_ENV = "CODEPIT_TINY_CHAT_GGUF_CACHE_PATH"
#: URL of the hosted FP16 *base* GGUF the lightweight path downloads then
#: quantizes locally (slice C / #274). Deliberately NOT the same as
#: TINY_CHAT_GGUF_URL_ENV (which submits a *finished* GGUF as-is): the
#: distinct `_BASE_FP16_` infix is the footgun guard. Defaults to the engine
#: route (#273) so a fresh agent needs zero config.
TINY_CHAT_BASE_FP16_URL_ENV = "CODEPIT_TINY_CHAT_BASE_FP16_URL"
TINY_CHAT_BASE_FP16_CACHE_PATH_ENV = "CODEPIT_TINY_CHAT_BASE_FP16_CACHE_PATH"
DEFAULT_TINY_CHAT_BASE_FP16_URL = (
    "https://engine.codepit.fun/v1/base-models/tiny-chat/fp16.gguf"
)
TINY_CHAT_ALLOWED_QUANTIZATION_PROFILES = ("q4_k_m", "q5_k_m", "q8_0")
CLIENT_SUBMISSION_ID_MAX_LENGTH = 128
MAX_UPLOAD_TTL_SECONDS = 60 * 60
UPLOAD_TTL_CLOCK_SKEW_SECONDS = 5
TERMINAL_SUBMISSION_STATES = frozenset(
    {
        "VALIDATION_FAILED",
        "BENCHMARK_FAILED",
        "VERIFIED",
        "SETTLED",
        "PUBLISHED",
        "CANCELLED",
        "INVALIDATED",
    },
)
RECEIPT_READY_SUBMISSION_STATES = frozenset({"VERIFIED", "SETTLED", "PUBLISHED"})


class OrchestratorError(RuntimeError):
    """Raised when the agent cannot complete the join → submit → verify flow."""


@dataclass
class OrchestratorConfig:
    base_url: str
    work_dir: Path
    source_model: str = DEFAULT_SOURCE_MODEL
    source_model_revision: str = DEFAULT_SOURCE_MODEL_REVISION
    private_key: str | None = None
    agent_wallet_private_key: str | None = None
    agent_id: str | None = None
    runtime_credential: str | None = None
    runtime_credential_id: str | None = None
    trust_tier: str | None = None
    challenge_id: str | None = None
    display_name: str | None = None
    declared_at_version: str = DEFAULT_DECLARED_AT_VERSION
    poll_interval_s: float = 3.0
    poll_timeout_s: float = 5 * 60.0
    receipt_poll_timeout_s: float | None = None
    request_timeout_s: float = 30.0
    session_path: Path | None = DEFAULT_SESSION_PATH
    pre_built_bundle_dir: Path | None = None
    skip_recipe_generation: bool = False
    selected_recipe: str | None = None
    client_submission_id: str | None = None
    capabilities: dict | None = None
    #: Optional LLM-backed Brain. When supplied, the orchestrator asks the
    #: Brain for a bounded optimization plan instead of only reordering fixed
    #: recipes. The Brain may either fall back to a safe default or raise in
    #: strict LLM-required mode; the orchestrator leaves that policy to the
    #: Brain config.
    brain: Brain | None = None
    #: Compact verifier-result history supplied to the Brain. The autonomous
    #: supervisor fills this from previous checked attempts so the next plan
    #: can react to what actually happened.
    brain_history: Sequence[Mapping[str, Any]] = field(default_factory=tuple)


@dataclass
class OrchestratorResult:
    agent_id: str
    signer_address: str
    challenge_id: str
    submission_id: str
    state: str
    benchmark_target_version: str
    chosen_recipe: str
    bundle_dir: Path
    proof_record_id: str | None = None
    settlement_ref: str | None = None
    upload_summary: Any | None = None
    balances: dict = field(default_factory=dict)
    rewards: dict = field(default_factory=dict)
    reused_session: bool = False
    recipe_failures: list[str] = field(default_factory=list)
    brain_decision: RecipeChoice | None = None
    brain_plan: OptimizationPlan | None = None
    client_submission_id: str | None = None
    result_id: str | None = None
    receipt_path: str | None = None
    public_result: dict | None = None
    baseline_comparison: dict | None = None
    verified_improvement: bool = False


@dataclass(frozen=True)
class PublicReceiptObservation:
    result_id: str
    receipt_path: str
    public_result: dict
    baseline_comparison: dict | None
    verified_improvement: bool


def _make_client(
    base_url: str,
    *,
    runtime_credential: str | None = None,
    request_timeout_s: float = 30.0,
) -> CodePitClient:
    kwargs: dict[str, Any] = {}
    if runtime_credential is not None:
        kwargs["credential"] = runtime_credential
    if request_timeout_s != 30.0:
        kwargs["timeout"] = request_timeout_s
    return CodePitClient(base_url=base_url, **kwargs)


def run_optimizer_agent(config: OrchestratorConfig) -> OrchestratorResult:
    _validate_selected_recipe(config.selected_recipe)
    if config.client_submission_id is not None:
        _validate_client_submission_id(config.client_submission_id)
    config.work_dir.mkdir(parents=True, exist_ok=True)

    signer, persisted_session = _resolve_signer(config)
    base_client = _make_client(config.base_url, request_timeout_s=config.request_timeout_s)
    session, reused = _ensure_session(config, base_client, signer, persisted_session)
    client = base_client.with_credentials(session.agent_id, session.runtime_credential)

    challenge_id = _resolve_challenge(client, config.challenge_id)
    challenge = client.read_challenge(challenge_id)
    benchmark_target_version = challenge.get("benchmark_target_version") or _challenge_target_version(challenge)
    if not benchmark_target_version:
        raise OrchestratorError(f"challenge {challenge_id} did not return a benchmark_target_version")

    eligibility = client.read_eligibility(challenge_id)
    if not eligibility.get("eligible"):
        reasons = eligibility.get("reasons") or ["unknown"]
        raise OrchestratorError(f"agent {session.agent_id} not eligible for {challenge_id}: {reasons}")

    chosen_dir, chosen_recipe, recipe_failures, brain_decision, brain_plan = _resolve_candidate_bundle_dir(
        config,
        challenge=challenge,
    )
    bundle_files = load_bundle(chosen_dir)
    optimization_notes = _build_optimization_notes(
        chosen_recipe=chosen_recipe,
        brain_plan=brain_plan,
        history=config.brain_history,
    )
    manifest_envelope = to_manifest_envelope(
        bundle_files,
        benchmark_target_version=benchmark_target_version,
        source_model_identifier=f"hf://{config.source_model}",
        source_model_revision=config.source_model_revision,
        optimization_methods=[chosen_recipe],
        optimization_notes=optimization_notes,
    )
    client_submission_id = _resolve_client_submission_id(
        config.client_submission_id,
        agent_id=session.agent_id,
        challenge_id=challenge_id,
        manifest_envelope=manifest_envelope,
    )

    submission_body = {
        "protocol_version": "v1",
        "client_submission_id": client_submission_id,
        "agent_id": session.agent_id,
        "challenge_id": challenge_id,
        "manifest_schema_version": benchmark_target_version,
        "manifest_envelope": manifest_envelope,
    }
    created = client.create_submission(submission_body)
    submission_id = created["submission_id"]
    _drive_uploads(client, created, bundle_files)

    terminal = _poll_to_terminal(
        client,
        submission_id,
        interval_s=config.poll_interval_s,
        timeout_s=config.poll_timeout_s,
    )
    terminal_state = str(terminal.get("state", "UNKNOWN"))
    receipt = None
    if terminal_state in RECEIPT_READY_SUBMISSION_STATES:
        receipt = _poll_public_receipt(
            client,
            submission_id,
            interval_s=config.poll_interval_s,
            timeout_s=config.receipt_poll_timeout_s or config.poll_timeout_s,
        )

    balances = _read_optional_agent_summary(client.read_balances)
    rewards = _read_optional_agent_summary(client.read_rewards)

    if config.session_path is not None and not reused:
        save_session(session, path=config.session_path)

    return OrchestratorResult(
        agent_id=session.agent_id,
        signer_address=session.signer_address,
        challenge_id=challenge_id,
        submission_id=submission_id,
        state=terminal_state,
        benchmark_target_version=benchmark_target_version,
        chosen_recipe=chosen_recipe,
        bundle_dir=chosen_dir,
        proof_record_id=terminal.get("proof_record_id"),
        settlement_ref=terminal.get("settlement_ref"),
        upload_summary=terminal.get("upload_summary"),
        balances=balances,
        rewards=rewards,
        reused_session=reused,
        recipe_failures=recipe_failures,
        brain_decision=brain_decision,
        brain_plan=brain_plan,
        client_submission_id=client_submission_id,
        result_id=receipt.result_id if receipt else None,
        receipt_path=receipt.receipt_path if receipt else None,
        public_result=receipt.public_result if receipt else None,
        baseline_comparison=receipt.baseline_comparison if receipt else None,
        verified_improvement=receipt.verified_improvement if receipt else False,
    )


# ---------------------------------------------------------------------------
# Long-running supervisor (run-forever)
# ---------------------------------------------------------------------------


_NO_CHALLENGE_MARKERS = ("no eligible challenge",)

ONNX_BROWSER_WEBGPU_ARTIFACT_LANE = "onnx-browser-webgpu"

# A lane runner is "the recipe path for one artifact lane". Each takes a
# fully-resolved OrchestratorConfig (the supervisor sets challenge_id to the
# peeked challenge id before invoking) and returns a terminal result. The
# registry below maps every lane the agent claims to know about to one of
# these — picking a runner is purely a function of the challenge's lane, not
# of any agent-side mode flag, so a single agent binary can serve any lane it
# has registered for. See #87 design.
LaneRunner = Callable[["OrchestratorConfig"], "OrchestratorResult"]


def default_lane_runners() -> dict[str, LaneRunner]:
    """Out-of-the-box lane registry shipped with the kit.

    Maps the two production lanes today (encoder + tiny-chat) to their
    respective recipe paths. A freshly-deployed agent — managed or external —
    picks up this default and can claim any eligible challenge on either lane
    without operator intervention. Extending to a new lane is a one-line
    registry addition here (or a per-deployment override on
    ``ForeverConfig.lane_runners``); the supervisor's dispatch loop stays
    unchanged.
    """
    return {
        ONNX_BROWSER_WEBGPU_ARTIFACT_LANE: run_optimizer_agent,
        OLLAMA_GGUF_LOCAL_ARTIFACT_LANE: run_tiny_chat_lane,
    }


def peek_next_eligible_challenge(
    config: "OrchestratorConfig",
) -> tuple[str, str] | None:
    """Return ``(challenge_id, artifact_lane)`` for the next challenge the
    engine would surface to this agent, or ``None`` when the network is idle.

    Used by the supervisor to choose a lane runner before claiming the
    challenge. The lookup is intentionally lightweight — one
    ``GET /v1/challenges/next`` followed by ``GET /v1/challenges/:id`` to read
    the lane. The matched runner re-targets the same challenge id by setting
    ``OrchestratorConfig.challenge_id`` so no second peek/claim race window
    exists between dispatch and execution.

    Raises ``OrchestratorError`` (with the canonical no-eligible-challenge
    marker) when ``/v1/challenges/next`` reports no work; this lets the
    supervisor's existing ``_NO_CHALLENGE_MARKERS`` path treat it as an idle
    tick rather than a fatal failure.
    """
    client = _make_client(
        base_url=config.base_url,
        runtime_credential=config.runtime_credential,
        request_timeout_s=config.request_timeout_s,
    )
    response = client.next_challenge()
    challenge = response.get("challenge")
    if not isinstance(challenge, Mapping) or not challenge.get("challenge_id"):
        return None
    challenge_id = str(challenge["challenge_id"])
    full = client.read_challenge(challenge_id)
    lane = full.get("artifact_lane")
    if not isinstance(lane, str) or not lane:
        # An open challenge missing artifact_lane is a server-side schema
        # invariant violation. Treat as no_challenge so the supervisor idles
        # rather than crashing, but log loud so operators see it.
        return None
    return challenge_id, lane


@dataclass
class ForeverConfig:
    """Configuration for ``run_optimizer_agent_forever``.

    The ``base_config`` is reused for every iteration; the wrapper only
    adjusts ``challenge_id`` (always None — pick the next eligible one)
    and creates a fresh per-iteration work subdirectory so candidates
    from earlier iterations don't shadow new ones.

    Lane dispatch (#87): set ``lane_runners`` to a registry of
    ``{artifact_lane -> LaneRunner}`` to enable peek-and-dispatch — every
    iteration polls the next eligible challenge, looks up the runner by the
    challenge's lane, and invokes it with the challenge id pre-targeted.
    When ``None`` (the default), the supervisor calls ``run_optimizer_agent``
    directly — preserves the legacy single-recipe behavior for tests and any
    operator still on the pre-#87 contract. Production entry points
    (``modal_app.py``, CLI ``run-forever``) opt in by passing
    ``default_lane_runners()``.
    """

    base_config: OrchestratorConfig
    idle_sleep_s: float = 30.0
    error_backoff_s: float = 15.0
    max_iterations: int | None = None
    max_runtime_s: float | None = None
    fresh_work_dir_per_iteration: bool = True
    on_iteration: Callable[[int, "ForeverIterationOutcome"], None] | None = None
    lane_runners: Mapping[str, LaneRunner] | None = None


@dataclass
class ForeverIterationOutcome:
    iteration: int
    started_at: float
    finished_at: float
    kind: str  # "result" | "no_challenge" | "transient_error" | "fatal_error"
    result: OrchestratorResult | None = None
    error_message: str | None = None
    error_code: str | None = None


@dataclass
class ForeverSummary:
    iterations_started: int = 0
    iterations_completed: int = 0
    terminal_state_counts: dict[str, int] = field(default_factory=dict)
    transient_error_count: int = 0
    last_result: OrchestratorResult | None = None
    last_error: str | None = None
    stopped_reason: str = "stop_event"


def run_optimizer_agent_forever(
    forever: ForeverConfig,
    *,
    stop_event: threading.Event | None = None,
    install_signal_handlers: bool = True,
) -> ForeverSummary:
    """Run the optimizer agent in a supervised loop until told to stop.

    The supervisor guarantees:
    - "no eligible challenge" responses sleep ``idle_sleep_s`` and retry,
      not crash. This is the steady-state when the network has no work.
    - retryable ``ProtocolError`` failures sleep ``error_backoff_s``.
      Non-retryable protocol errors stop the loop with ``fatal_error``.
    - SIGTERM and SIGINT (Ctrl-C) flip the internal stop_event so an
      in-flight iteration completes its final poll before exiting. This
      matches how a process supervisor (systemd, k8s, Modal) expects a
      worker to behave.
    """
    summary = ForeverSummary()
    started_wall = time.monotonic()
    brain_history: list[dict[str, Any]] = [
        _compact_history_entry(item) for item in forever.base_config.brain_history
    ][-8:]

    if stop_event is None:
        stop_event = threading.Event()

    if install_signal_handlers:
        _install_stop_signals(stop_event)

    iteration = 0
    while not stop_event.is_set():
        if forever.max_iterations is not None and iteration >= forever.max_iterations:
            summary.stopped_reason = "max_iterations"
            break
        if forever.max_runtime_s is not None and (
            time.monotonic() - started_wall >= forever.max_runtime_s
        ):
            summary.stopped_reason = "max_runtime"
            break

        iteration += 1
        summary.iterations_started += 1
        outcome = _run_one_iteration(forever, iteration, brain_history=brain_history)

        if forever.on_iteration is not None:
            try:
                forever.on_iteration(iteration, outcome)
            except Exception:
                # Per-iteration callback failures must not take down the loop.
                pass

        if outcome.kind == "result":
            summary.iterations_completed += 1
            assert outcome.result is not None
            summary.last_result = outcome.result
            state = outcome.result.state
            summary.terminal_state_counts[state] = (
                summary.terminal_state_counts.get(state, 0) + 1
            )
            brain_history.append(_summarize_result_for_brain(outcome.result))
            del brain_history[:-8]
            # Loop on; the agent may pick up another challenge immediately.
            continue

        if outcome.kind == "no_challenge":
            stop_event.wait(forever.idle_sleep_s)
            continue

        if outcome.kind == "transient_error":
            summary.transient_error_count += 1
            summary.last_error = outcome.error_message
            stop_event.wait(forever.error_backoff_s)
            continue

        if outcome.kind == "fatal_error":
            summary.last_error = outcome.error_message
            summary.stopped_reason = "fatal_error"
            return summary

    return summary


def _run_one_iteration(
    forever: ForeverConfig,
    iteration: int,
    *,
    brain_history: Sequence[Mapping[str, Any]] = (),
) -> ForeverIterationOutcome:
    started_at = time.monotonic()
    base = forever.base_config
    work_dir = (
        base.work_dir / f"iter-{iteration:06d}"
        if forever.fresh_work_dir_per_iteration
        else base.work_dir
    )
    iteration_config = OrchestratorConfig(
        base_url=base.base_url,
        work_dir=work_dir,
        source_model=base.source_model,
        source_model_revision=base.source_model_revision,
        private_key=base.private_key,
        agent_wallet_private_key=base.agent_wallet_private_key,
        agent_id=base.agent_id,
        runtime_credential=base.runtime_credential,
        runtime_credential_id=base.runtime_credential_id,
        trust_tier=base.trust_tier,
        challenge_id=None,  # always pick the next eligible challenge
        display_name=base.display_name,
        declared_at_version=base.declared_at_version,
        poll_interval_s=base.poll_interval_s,
        poll_timeout_s=base.poll_timeout_s,
        receipt_poll_timeout_s=base.receipt_poll_timeout_s,
        request_timeout_s=base.request_timeout_s,
        session_path=base.session_path,
        pre_built_bundle_dir=base.pre_built_bundle_dir,
        skip_recipe_generation=base.skip_recipe_generation,
        selected_recipe=base.selected_recipe,
        client_submission_id=base.client_submission_id,
        capabilities=base.capabilities,
        brain=base.brain,
        brain_history=tuple(_compact_history_entry(item) for item in brain_history),
    )

    try:
        if forever.lane_runners is not None:
            peeked = peek_next_eligible_challenge(iteration_config)
            if peeked is None:
                return ForeverIterationOutcome(
                    iteration=iteration,
                    started_at=started_at,
                    finished_at=time.monotonic(),
                    kind="no_challenge",
                    error_message="peek_next_eligible_challenge returned None",
                )
            challenge_id, lane = peeked
            runner = forever.lane_runners.get(lane)
            if runner is None:
                # An eligible challenge exists but on a lane this agent has
                # no runner for. Skip the iteration rather than crash — keeps
                # an agent that knows lane A safe against a future network
                # that adds lane B (#87 "robust generic" property). Operators
                # can extend coverage by registering the missing runner.
                return ForeverIterationOutcome(
                    iteration=iteration,
                    started_at=started_at,
                    finished_at=time.monotonic(),
                    kind="no_challenge",
                    error_message=(
                        f"no runner registered for artifact_lane {lane!r} "
                        f"(challenge {challenge_id}); skipping"
                    ),
                )
            iteration_config = _with_challenge_id(iteration_config, challenge_id)
            result = runner(iteration_config)
        else:
            # Legacy direct-call path (lane_runners not configured). The
            # runner is responsible for polling and lane selection itself —
            # preserves the pre-#87 contract that the existing forever-loop
            # tests in `test_run_forever.py` exercise.
            result = run_optimizer_agent(iteration_config)
        return ForeverIterationOutcome(
            iteration=iteration,
            started_at=started_at,
            finished_at=time.monotonic(),
            kind="result",
            result=result,
        )
    except OrchestratorError as error:
        message = str(error)
        if any(marker in message.lower() for marker in _NO_CHALLENGE_MARKERS):
            return ForeverIterationOutcome(
                iteration=iteration,
                started_at=started_at,
                finished_at=time.monotonic(),
                kind="no_challenge",
                error_message=message,
            )
        return ForeverIterationOutcome(
            iteration=iteration,
            started_at=started_at,
            finished_at=time.monotonic(),
            kind="transient_error",
            error_message=message,
        )
    except ProtocolError as error:
        kind = "transient_error" if (error.retryable is None or error.retryable) else "fatal_error"
        return ForeverIterationOutcome(
            iteration=iteration,
            started_at=started_at,
            finished_at=time.monotonic(),
            kind=kind,
            error_message=str(error),
            error_code=error.code,
        )
    except Exception as error:  # pragma: no cover - defensive
        return ForeverIterationOutcome(
            iteration=iteration,
            started_at=started_at,
            finished_at=time.monotonic(),
            kind="transient_error",
            error_message=f"{type(error).__name__}: {error}",
        )


def _install_stop_signals(stop_event: threading.Event) -> None:
    def _handler(_signum: int, _frame: Any) -> None:
        stop_event.set()

    try:
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):
        # signal.signal raises if not on the main thread or unsupported
        # platform. Tests pass their own stop_event so this is best-effort.
        pass


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def register_or_load_external_agent(
    *,
    base_url: str,
    session_path: Path = DEFAULT_SESSION_PATH,
    display_name: str | None = None,
    capabilities: dict | None = None,
) -> tuple[AgentSession, bool]:
    """Register a fresh external agent (or reuse the persisted one) without
    running the full optimizer loop.

    Returns ``(session, reused)``. The session is persisted to ``session_path``
    so subsequent CLI commands (``modelbook-run``, ``run``) can pick it up
    without further credential plumbing.

    Capabilities default to the live ``tiny-chat`` lane so the documented
    ``register-external`` -> ``modelbook-run --submit`` path is eligible for the
    open challenge without extra flags (issue #258). Callers may pass explicit
    ``capabilities`` (e.g. from :func:`resolve_register_lane_capabilities`) to
    target a different lane.
    """

    from pathlib import Path as _Path

    config = OrchestratorConfig(
        base_url=base_url,
        work_dir=_Path(".cache/codepit-register"),
        session_path=session_path,
        display_name=display_name,
        capabilities=capabilities
        or _default_tiny_chat_capabilities(DEFAULT_DECLARED_AT_VERSION),
    )
    signer, persisted = _resolve_signer(config)
    client = CodePitClient(config.base_url)
    session, reused = _ensure_session(config, client, signer, persisted)
    # _ensure_session only saves on the explicit-credentials path; the
    # fresh-registration branch returns without persisting. Save here so
    # subsequent CLI invocations find the credential.
    if not reused:
        save_session(session, session_path)
    return session, reused


# ---------------------------------------------------------------------------
# Tiny-chat (ollama-gguf-local) external-agent run path
# ---------------------------------------------------------------------------


@dataclass
class TinyChatRunConfig:
    """Configuration for a single external-agent tiny-chat submission run.

    Mirrors the auth/session/poll knobs of :class:`OrchestratorConfig` but
    swaps ONNX recipe generation for a real GGUF build (``gguf_build`` seam,
    default the env-gated llama.cpp pipeline of #49) plus the tiny-chat bundle
    assembler. The run drives the same canonical protocol loop via the shared
    orchestrator helpers — there is no parallel auth or submit path.
    """

    base_url: str
    work_dir: Path
    #: Explicit ollama-gguf-local challenge id. When None, targeting falls back
    #: to ``target`` and the resolved challenge's lane is verified before any
    #: compute is spent.
    challenge_id: str | None = None
    #: Challenge targeting when ``challenge_id`` is None. ``"sponsor"`` discovers
    #: the richest eligible open sponsor competition (slice G); ``"auto"`` /
    #: ``"bootstrap"`` use the shared ``/v1/challenges/next`` discovery.
    target: str = "auto"
    #: Escape hatch for the payout-binding guard (slice F). When False (default),
    #: running a rewarded sponsor target with no bound payout address is refused
    #: so a verified reward can't be silently forfeited.
    allow_unbound_payout: bool = False
    base_model_ref: str = "hf://Qwen/Qwen2.5-0.5B-Instruct"
    source_model_revision: str = "main"
    quantization_profile: str = "q4_k_m"
    optimization_methods: list[str] | None = None
    private_key: str | None = None
    agent_wallet_private_key: str | None = None
    agent_id: str | None = None
    runtime_credential: str | None = None
    runtime_credential_id: str | None = None
    trust_tier: str | None = None
    display_name: str | None = None
    declared_at_version: str = DEFAULT_DECLARED_AT_VERSION
    poll_interval_s: float = 3.0
    poll_timeout_s: float = 5 * 60.0
    receipt_poll_timeout_s: float | None = None
    request_timeout_s: float = 30.0
    session_path: Path | None = DEFAULT_SESSION_PATH
    client_submission_id: str | None = None
    capabilities: dict | None = None
    #: GGUF build seam: ``(*, base_model_ref, quantization_profile, out_gguf) ->
    #: provenance``. Defaults to the env-gated real builder; if neither this nor
    #: the env toolchain is configured the run aborts rather than ship a
    #: non-real binary.
    gguf_build: Callable[..., dict[str, Any]] | None = None
    #: A real GGUF the agent already built with its own toolchain. When set, it
    #: is submitted as-is and ``gguf_build`` is skipped. Validated to be a real
    #: GGUF binary (magic header) before submission.
    gguf_path: Path | None = None
    #: Optional LLM-backed Brain. In the tiny-chat lane this is deliberately
    #: bounded to choosing supported GGUF quantization profiles.
    brain: Brain | None = None
    brain_history: Sequence[Mapping[str, Any]] = field(default_factory=tuple)


def run_tiny_chat_external_agent(config: TinyChatRunConfig) -> OrchestratorResult:
    """Register (or reuse), build a real GGUF, assemble the ollama-gguf-local
    bundle, submit over the canonical protocol, and poll to a terminal result.
    """
    if config.client_submission_id is not None:
        _validate_client_submission_id(config.client_submission_id)
    config.work_dir.mkdir(parents=True, exist_ok=True)

    session_config = _tiny_chat_session_config(config)
    signer, persisted_session = _resolve_signer(session_config)
    base_client = _make_client(config.base_url, request_timeout_s=config.request_timeout_s)
    session, reused = _ensure_session(session_config, base_client, signer, persisted_session)
    client = base_client.with_credentials(session.agent_id, session.runtime_credential)

    challenge_id = _resolve_tiny_chat_challenge(client, config)
    challenge = client.read_challenge(challenge_id)
    artifact_lane = challenge.get("artifact_lane")
    if artifact_lane != OLLAMA_GGUF_LOCAL_ARTIFACT_LANE:
        raise OrchestratorError(
            f"challenge {challenge_id} is artifact_lane {artifact_lane!r}; "
            f"the tiny-chat run requires '{OLLAMA_GGUF_LOCAL_ARTIFACT_LANE}'",
        )
    benchmark_target_version = challenge.get("benchmark_target_version") or _challenge_target_version(challenge)
    if not benchmark_target_version:
        raise OrchestratorError(f"challenge {challenge_id} did not return a benchmark_target_version")

    eligibility = client.read_eligibility(challenge_id)
    if not eligibility.get("eligible"):
        reasons = eligibility.get("reasons") or ["unknown"]
        raise OrchestratorError(f"agent {session.agent_id} not eligible for {challenge_id}: {reasons}")

    # Fail closed before spending compute: a rewarded sponsor target with no
    # bound payout address would forfeit the reward on settlement (slice F).
    _assert_payout_bound_for_reward(client, config)

    quantization_profile = config.quantization_profile
    if config.brain is not None:
        allowed_profiles = (
            (config.quantization_profile,)
            if config.gguf_path is not None
            else TINY_CHAT_ALLOWED_QUANTIZATION_PROFILES
        )
        choice = config.brain.pick_tiny_chat_quantization(
            challenge_spec=challenge,
            allowed_profiles=allowed_profiles,
            current_profile=config.quantization_profile,
            history=config.brain_history,
        )
        quantization_profile = choice.quantization_profile

    if config.gguf_path is not None:
        # The agent built the GGUF out-of-band with its own toolchain.
        if not config.gguf_path.is_file():
            raise OrchestratorError(f"gguf_path does not exist: {config.gguf_path}")
        gguf_bytes = config.gguf_path.read_bytes()
        if gguf_bytes[:4] != b"GGUF":
            raise OrchestratorError(
                f"gguf_path {config.gguf_path} is not a GGUF binary (bad magic); "
                "the tiny-chat lane requires a real GGUF, never a fixture.",
            )
    else:
        builder = config.gguf_build or _resolve_default_gguf_builder()
        if builder is None:
            raise OrchestratorError(
                "no GGUF builder configured; set the CODEPIT_GGUF_* llama.cpp env (see #49), "
                "pass TinyChatRunConfig.gguf_build, or supply a pre-built TinyChatRunConfig.gguf_path. "
                "The tiny-chat lane never ships a fixture.",
            )
        gguf_path = config.work_dir / f"tiny-chat-{_safe_filename(quantization_profile)}.gguf"
        builder(
            base_model_ref=config.base_model_ref,
            quantization_profile=quantization_profile,
            out_gguf=gguf_path,
        )
        gguf_bytes = gguf_path.read_bytes()

    bundle_files = build_tiny_chat_bundle_files(
        gguf_bytes=gguf_bytes,
        base_model_ref=config.base_model_ref,
        quantization_profile=quantization_profile,
    )
    manifest_envelope = to_tiny_chat_manifest_envelope(
        bundle_files,
        benchmark_target_version=benchmark_target_version,
        base_model_ref=config.base_model_ref,
        source_model_revision=config.source_model_revision,
        quantization_profile=quantization_profile,
        optimization_methods=config.optimization_methods or [quantization_profile],
    )
    client_submission_id = _resolve_client_submission_id(
        config.client_submission_id,
        agent_id=session.agent_id,
        challenge_id=challenge_id,
        manifest_envelope=manifest_envelope,
    )

    submission_body = {
        "protocol_version": "v1",
        "client_submission_id": client_submission_id,
        "agent_id": session.agent_id,
        "challenge_id": challenge_id,
        "manifest_schema_version": benchmark_target_version,
        "manifest_envelope": manifest_envelope,
    }
    created = client.create_submission(submission_body)
    submission_id = created["submission_id"]
    _drive_uploads(client, created, bundle_files)

    terminal = _poll_to_terminal(
        client,
        submission_id,
        interval_s=config.poll_interval_s,
        timeout_s=config.poll_timeout_s,
    )
    terminal_state = str(terminal.get("state", "UNKNOWN"))
    receipt = None
    if terminal_state in RECEIPT_READY_SUBMISSION_STATES:
        receipt = _poll_public_receipt(
            client,
            submission_id,
            interval_s=config.poll_interval_s,
            timeout_s=config.receipt_poll_timeout_s or config.poll_timeout_s,
        )

    balances = _read_optional_agent_summary(client.read_balances)
    rewards = _read_optional_agent_summary(client.read_rewards)

    if config.session_path is not None and not reused:
        save_session(session, path=config.session_path)

    return OrchestratorResult(
        agent_id=session.agent_id,
        signer_address=session.signer_address,
        challenge_id=challenge_id,
        submission_id=submission_id,
        state=terminal_state,
        benchmark_target_version=benchmark_target_version,
        chosen_recipe=quantization_profile,
        bundle_dir=config.work_dir,
        proof_record_id=terminal.get("proof_record_id"),
        settlement_ref=terminal.get("settlement_ref"),
        upload_summary=terminal.get("upload_summary"),
        balances=balances,
        rewards=rewards,
        reused_session=reused,
        client_submission_id=client_submission_id,
        result_id=receipt.result_id if receipt else None,
        receipt_path=receipt.receipt_path if receipt else None,
        public_result=receipt.public_result if receipt else None,
        baseline_comparison=receipt.baseline_comparison if receipt else None,
        verified_improvement=receipt.verified_improvement if receipt else False,
    )


def _tiny_chat_session_config(config: TinyChatRunConfig) -> OrchestratorConfig:
    """Project a TinyChatRunConfig onto the OrchestratorConfig fields the shared
    signer/session helpers read, declaring tiny-chat capabilities by default.
    """
    return OrchestratorConfig(
        base_url=config.base_url,
        work_dir=config.work_dir,
        private_key=config.private_key,
        agent_wallet_private_key=config.agent_wallet_private_key,
        agent_id=config.agent_id,
        runtime_credential=config.runtime_credential,
        runtime_credential_id=config.runtime_credential_id,
        trust_tier=config.trust_tier,
        display_name=config.display_name,
        declared_at_version=config.declared_at_version,
        request_timeout_s=config.request_timeout_s,
        session_path=config.session_path,
        capabilities=config.capabilities or _default_tiny_chat_capabilities(config.declared_at_version),
    )


def _default_tiny_chat_capabilities(declared_at_version: str) -> dict:
    return {
        "declared_artifact_lanes": [OLLAMA_GGUF_LOCAL_ARTIFACT_LANE],
        "declared_at_version": declared_at_version,
        "declared_model_classes": ["chat-causal-small"],
        "declared_runtimes": ["ollama"],
        "optimization_methods": ["quantization"],
    }


#: Registration lanes a fresh external agent can declare. The default is the
#: live ``tiny-chat`` (``ollama-gguf-local`` / ``chat-causal-small``) lane so the
#: documented ``register-external`` -> ``modelbook-run --submit`` path matches the
#: open challenge out of the box (issue #258). ``onnx-encoder`` preserves the
#: legacy ONNX browser/WebGPU encoder lane for callers that target it.
REGISTER_LANES: tuple[str, ...] = ("tiny-chat", "onnx-encoder")
DEFAULT_REGISTER_LANE = "tiny-chat"


def resolve_register_lane_capabilities(
    lane: str = DEFAULT_REGISTER_LANE,
    declared_at_version: str = DEFAULT_DECLARED_AT_VERSION,
) -> dict:
    """Map a registration lane name to its declared capabilities.

    Capabilities are signed into the registration payload hash and cannot be
    changed after registration, so the lane must be chosen at register time.
    """

    if lane == "tiny-chat":
        return _default_tiny_chat_capabilities(declared_at_version)
    if lane == "onnx-encoder":
        return _default_capabilities(declared_at_version)
    raise ValueError(
        f"unknown registration lane {lane!r}; expected one of {REGISTER_LANES}"
    )


def run_tiny_chat_lane(config: OrchestratorConfig) -> OrchestratorResult:
    """Adapt the generic lane-runner config to the tiny-chat GGUF runner.

    The managed worker and forever supervisor both dispatch lane runners with an
    OrchestratorConfig. The tiny-chat implementation has its own config object
    because it carries GGUF-specific options, so the default registry must bridge
    the common fields before invoking it.
    """

    return run_tiny_chat_external_agent(
        TinyChatRunConfig(
            base_url=config.base_url,
            work_dir=config.work_dir,
            challenge_id=config.challenge_id,
            source_model_revision=config.source_model_revision,
            private_key=config.private_key,
            agent_wallet_private_key=config.agent_wallet_private_key,
            agent_id=config.agent_id,
            runtime_credential=config.runtime_credential,
            runtime_credential_id=config.runtime_credential_id,
            trust_tier=config.trust_tier,
            display_name=config.display_name,
            declared_at_version=config.declared_at_version,
            poll_interval_s=config.poll_interval_s,
            poll_timeout_s=config.poll_timeout_s,
            receipt_poll_timeout_s=config.receipt_poll_timeout_s,
            request_timeout_s=config.request_timeout_s,
            session_path=config.session_path,
            client_submission_id=config.client_submission_id,
            capabilities=config.capabilities,
            gguf_path=_resolve_env_tiny_chat_gguf_path(config.work_dir),
            brain=config.brain,
            brain_history=config.brain_history,
        )
    )


def _resolve_env_tiny_chat_gguf_path(work_dir: Path) -> Path | None:
    explicit_path = os.environ.get(TINY_CHAT_GGUF_PATH_ENV)
    if explicit_path:
        return Path(explicit_path)

    url = os.environ.get(TINY_CHAT_GGUF_URL_ENV)
    if not url:
        return None
    if not url.startswith("https://"):
        raise OrchestratorError(f"{TINY_CHAT_GGUF_URL_ENV} must be an https URL")

    cache_path = Path(
        os.environ.get(TINY_CHAT_GGUF_CACHE_PATH_ENV)
        or work_dir.parent / "tiny-chat-env.gguf"
    )
    if cache_path.exists():
        _validate_tiny_chat_gguf_file(cache_path)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.part")
    request = Request(url, headers={"user-agent": "codepit-model-optimizer/0.1"})
    try:
        with urlopen(request, timeout=60) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception as error:
        tmp_path.unlink(missing_ok=True)
        raise OrchestratorError(f"failed to download {TINY_CHAT_GGUF_URL_ENV}") from error

    _validate_tiny_chat_gguf_file(tmp_path)
    tmp_path.replace(cache_path)
    return cache_path


def _validate_tiny_chat_gguf_file(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            header = handle.read(4)
    except OSError as error:
        raise OrchestratorError(f"cannot read GGUF file {path}: {error}") from error
    if header != b"GGUF":
        raise OrchestratorError(f"GGUF file {path} has invalid magic header")


def _download_base_fp16_gguf(url: str, cache_path: Path) -> Path:
    """Download the hosted FP16 base GGUF (https only), caching atomically.

    Per-URL cache key avoids collisions across different bases. Validates the
    GGUF magic before use. Follows redirects (the engine route 302s to a
    presigned R2 URL), which urllib does by default.
    """
    if not url.startswith("https://"):
        raise OrchestratorError(f"{TINY_CHAT_BASE_FP16_URL_ENV} must be an https URL")
    if cache_path.exists():
        _validate_tiny_chat_gguf_file(cache_path)
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_name(f"{cache_path.name}.part")
    request = Request(url, headers={"user-agent": "codepit-model-optimizer/0.1"})
    try:
        with urlopen(request, timeout=120) as response, tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
    except Exception as error:
        tmp_path.unlink(missing_ok=True)
        raise OrchestratorError(f"failed to download {TINY_CHAT_BASE_FP16_URL_ENV}") from error

    _validate_tiny_chat_gguf_file(tmp_path)
    tmp_path.replace(cache_path)
    return cache_path


def _hosted_base_cache_path(work_dir: Path, url: str) -> Path:
    explicit = os.environ.get(TINY_CHAT_BASE_FP16_CACHE_PATH_ENV)
    if explicit:
        return Path(explicit)
    # Per-URL cache key so different base URLs don't collide on one filename.
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return work_dir.parent / f"tiny-chat-base-fp16-{digest}.gguf"


def _resolve_default_gguf_builder() -> Callable[..., dict[str, Any]] | None:
    # Lazy import: the pipeline pulls in subprocess/toolchain glue only needed
    # when an agent actually builds on its own compute.
    from .gguf_build_pipeline import (
        build_hosted_base_quantized_gguf,
        make_env_gguf_builder,
        resolve_quantize_bin,
    )

    # Full local convert+quantize toolchain wins when its env is set.
    env_builder = make_env_gguf_builder()
    if env_builder is not None:
        return env_builder

    # Lightweight fallback: download the hosted FP16 base and quantize-only.
    base_url = os.environ.get(TINY_CHAT_BASE_FP16_URL_ENV, DEFAULT_TINY_CHAT_BASE_FP16_URL)

    def builder(*, base_model_ref: str, quantization_profile: str, out_gguf: Path) -> dict[str, Any]:
        # Resolve the quantize binary first so a missing toolchain fails closed
        # with the one-line fix BEFORE any download work.
        quantize_bin = resolve_quantize_bin()
        cache_path = _hosted_base_cache_path(out_gguf.parent, base_url)

        def base_provider(_dest: Path) -> Path:
            return _download_base_fp16_gguf(base_url, cache_path)

        provenance = build_hosted_base_quantized_gguf(
            base_provider=base_provider,
            quantize_bin=quantize_bin,
            quantization_profile=quantization_profile,
            out_gguf=out_gguf,
        )
        provenance["base_model_ref"] = base_model_ref
        provenance["base_url"] = base_url
        return provenance

    return builder


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-.")
    return cleaned or "artifact"


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _resolve_signer(config: OrchestratorConfig) -> tuple[AgentSigner, AgentSession | None]:
    if config.private_key:
        return AgentSigner.from_private_key(config.private_key), None

    persisted = None
    if config.session_path is not None:
        persisted = load_session(config.session_path)
        if persisted is not None and persisted.base_url == config.base_url:
            return AgentSigner.from_private_key(persisted.signer_private_key), persisted

    return AgentSigner.ephemeral(), persisted


def _resolve_agent_wallet(
    config: OrchestratorConfig,
    persisted: AgentSession | None,
) -> AgentWallet:
    if config.agent_wallet_private_key:
        return AgentWallet.from_private_key(config.agent_wallet_private_key)
    if persisted is not None and persisted.agent_wallet_private_key:
        return AgentWallet.from_private_key(persisted.agent_wallet_private_key)
    return AgentWallet.ephemeral()


def _ensure_session(
    config: OrchestratorConfig,
    client: CodePitClient,
    signer: AgentSigner,
    persisted: AgentSession | None,
) -> tuple[AgentSession, bool]:
    if bool(config.agent_id) != bool(config.runtime_credential):
        raise OrchestratorError(
            "agent_id and runtime_credential must be provided together",
        )

    if config.agent_id and config.runtime_credential:
        wallet = _resolve_agent_wallet(config, persisted)
        session = AgentSession(
            base_url=config.base_url,
            agent_id=config.agent_id,
            signer_private_key=signer.private_key,
            signer_address=signer.address,
            runtime_credential=config.runtime_credential,
            runtime_credential_id=config.runtime_credential_id,
            trust_tier=config.trust_tier,
            agent_wallet_private_key=wallet.private_key,
            agent_wallet_address=wallet.address,
        )
        if config.session_path is not None:
            save_session(session, config.session_path)
        return session, True

    if persisted is not None and persisted.base_url == config.base_url and persisted.signer_address == signer.address:
        return persisted, True

    capabilities = config.capabilities or _default_capabilities(config.declared_at_version)
    display_name = config.display_name or f"{DEFAULT_DISPLAY_NAME_PREFIX}.{signer.address[2:8]}"
    agent_wallet = _resolve_agent_wallet(config, persisted)
    agent_wallet_payload = {
        "address": agent_wallet.address,
        "chain_id": 84532,
        "network": "base-sepolia",
        "wallet_provider": "local",
        "custody_mode": "agent_local",
    }
    normalized_payload = {
        "protocol_version": "v1",
        "agent_signer_address": signer.address.lower(),
        "agent": {"display_name": display_name, "mode": "external"},
        "capabilities": capabilities,
        "agent_wallet": agent_wallet_payload,
    }
    registration_payload_hash = hash_registration_payload(normalized_payload)

    challenge = client.request_auth_challenge(
        {
            "protocol_version": "v1",
            "agent_signer_address": signer.address,
            "registration_payload_hash": registration_payload_hash,
        },
    )
    signature = signer.sign_message(challenge["message"])
    wallet_timestamp_ms = int(time.time() * 1000)
    agent_wallet_signature = agent_wallet.sign_message(
        build_agent_wallet_binding_message(
            agent_signer_address=signer.address,
            agent_wallet_address=agent_wallet.address,
            registration_payload_hash=registration_payload_hash,
            timestamp_ms=wallet_timestamp_ms,
        )
    )

    register_body = {
        "protocol_version": "v1",
        "challenge_id": challenge["challenge_id"],
        "nonce": challenge["nonce"],
        "timestamp_ms": int(time.time() * 1000),
        "signature": signature,
        "agent_signer_address": signer.address,
        "agent": normalized_payload["agent"],
        "capabilities": capabilities,
        "agent_wallet": agent_wallet_payload,
        "agent_wallet_signature": agent_wallet_signature,
        "agent_wallet_timestamp_ms": wallet_timestamp_ms,
    }
    sybil_gate_solution = _solve_registration_sybil_gate(
        challenge.get("sybil_gate"),
        signer_address=signer.address,
        registration_payload_hash=registration_payload_hash,
        challenge_nonce=challenge["nonce"],
    )
    if sybil_gate_solution is not None:
        register_body["sybil_gate_solution"] = sybil_gate_solution
    registered = client.register(register_body)
    credential = registered["credential"]
    return (
        AgentSession(
            base_url=config.base_url,
            agent_id=registered["agent_id"],
            signer_private_key=signer.private_key,
            signer_address=signer.address,
            runtime_credential=credential["secret"],
            runtime_credential_id=credential.get("id"),
            trust_tier=registered.get("trust_tier"),
            agent_wallet_private_key=agent_wallet.private_key,
            agent_wallet_address=agent_wallet.address,
        ),
        False,
    )


def _solve_registration_sybil_gate(
    gate: Any,
    *,
    signer_address: str,
    registration_payload_hash: str,
    challenge_nonce: str,
) -> dict[str, str] | None:
    if gate is None:
        return None
    if not isinstance(gate, Mapping):
        raise OrchestratorError("unsupported registration sybil gate payload")
    if gate.get("kind") != "hashcash":
        raise OrchestratorError(f"unsupported registration sybil gate kind: {gate.get('kind')!r}")

    try:
        difficulty_bits = int(gate.get("difficulty_bits", 0))
    except (TypeError, ValueError) as exc:
        raise OrchestratorError("registration sybil gate difficulty must be an integer") from exc
    difficulty_bits = max(0, min(256, difficulty_bits))

    solution_nonce = 0
    while True:
        nonce = str(solution_nonce)
        digest = _registration_pow_digest(
            signer_address=signer_address,
            registration_payload_hash=registration_payload_hash,
            challenge_nonce=challenge_nonce,
            solution_nonce=nonce,
        )
        if _count_leading_zero_bits(digest) >= difficulty_bits:
            return {"kind": "hashcash", "nonce": nonce}
        solution_nonce += 1


def _registration_pow_digest(
    *,
    signer_address: str,
    registration_payload_hash: str,
    challenge_nonce: str,
    solution_nonce: str,
) -> str:
    return hashlib.sha256(
        ":".join(
            [
                REGISTRATION_POW_INPUT_PREFIX,
                signer_address.lower(),
                registration_payload_hash,
                challenge_nonce,
                solution_nonce,
            ]
        ).encode("utf-8")
    ).hexdigest()


def _count_leading_zero_bits(hex_digest: str) -> int:
    bits = 0
    for char in hex_digest:
        value = int(char, 16)
        if value == 0:
            bits += 4
            continue
        for mask in (8, 4, 2, 1):
            if value & mask:
                return bits
            bits += 1
    return bits


def _with_challenge_id(
    config: "OrchestratorConfig", challenge_id: str
) -> "OrchestratorConfig":
    """Return a copy of ``config`` targeting a specific challenge id.

    Used by the supervisor after lane peek-and-dispatch so the chosen runner
    re-fetches exactly the challenge whose lane drove the dispatch decision
    — no second poll, no race window between peek and run.
    """
    return replace(config, challenge_id=challenge_id)


def _resolve_challenge(client: CodePitClient, explicit_challenge_id: str | None) -> str:
    if explicit_challenge_id:
        return explicit_challenge_id
    response = client.next_challenge()
    challenge = response.get("challenge")
    if not isinstance(challenge, Mapping) or not challenge.get("challenge_id"):
        raise OrchestratorError("no eligible challenge returned by /v1/challenges/next")
    return str(challenge["challenge_id"])


def _assert_payout_bound_for_reward(client: CodePitClient, config: "TinyChatRunConfig") -> None:
    """Refuse to run a rewarded sponsor target with no bound payout address.

    A verified sponsor submission whose agent has no ``payout_address`` silently
    forfeits its reward (settlement marks ``missing_payout_address`` and skips
    allocation). Rather than let an agent spend compute to earn a reward it
    cannot receive, fail closed and tell the operator how to bind a wallet via
    the owner-claim flow (slice F, #272). ``--allow-unbound-payout`` overrides.
    """
    if config.target != "sponsor" or config.allow_unbound_payout:
        return
    agent = client.read_agent()
    payout_address = agent.get("payout_address") if isinstance(agent, Mapping) else None
    if not payout_address:
        raise OrchestratorError(
            "refusing to run a rewarded sponsor target with no bound payout "
            "address: a verified reward would be forfeited. Bind a payout wallet "
            "via the owner claim flow ('codepit-model-optimizer claim-agent ...'), "
            "or pass --allow-unbound-payout to proceed anyway.",
        )


def _resolve_tiny_chat_challenge(client: CodePitClient, config: "TinyChatRunConfig") -> str:
    """Resolve which ollama-gguf-local challenge the tiny-chat run targets.

    Precedence: an explicit ``challenge_id`` always wins (operator pinned it).
    Otherwise ``target == "sponsor"`` discovers the richest eligible open
    sponsor competition so the agent enters a *rewarded* challenge instead of
    relying on bootstrap luck (slice G, #276). Any other target falls back to
    the shared ``/v1/challenges/next`` discovery, unchanged.
    """
    if config.challenge_id:
        return config.challenge_id
    if config.target == "sponsor":
        challenge_id = discover_sponsor_challenge(
            client,
            artifact_lane=OLLAMA_GGUF_LOCAL_ARTIFACT_LANE,
        )
        if challenge_id is None:
            raise OrchestratorError(
                "no eligible open sponsor competition found for --target sponsor",
            )
        return challenge_id
    return _resolve_challenge(client, None)


def _challenge_target_version(challenge: Mapping[str, Any]) -> str | None:
    target = challenge.get("benchmark_target")
    if isinstance(target, Mapping):
        version = target.get("version")
        if isinstance(version, str):
            return version
    return None


def _build_optimization_notes(
    *,
    chosen_recipe: str,
    brain_plan: OptimizationPlan | None,
    history: Sequence[Mapping[str, Any]],
) -> str:
    if brain_plan is None:
        return f"codepit-model-optimizer recipe={chosen_recipe}"

    first_experiment = brain_plan.experiments[0] if brain_plan.experiments else None
    parts = [f"Brain plan: {brain_plan.strategy.strip()}"]
    if first_experiment is not None:
        parts.append(
            f"First experiment: {first_experiment.name} - {first_experiment.hypothesis.strip()}",
        )
    if history:
        parts.append(
            f"Used {min(len(history), 8)} prior checked attempt(s) to choose this run.",
        )
    parts.append(f"Submitted recipe: {chosen_recipe}.")
    return " ".join(_truncate_text(part, 260) for part in parts if part).strip()


def _summarize_result_for_brain(result: OrchestratorResult) -> dict[str, Any]:
    public_metrics = (
        result.public_result.get("metrics")
        if isinstance(result.public_result, Mapping)
        else None
    )
    comparison = result.baseline_comparison if isinstance(result.baseline_comparison, Mapping) else {}
    summary: dict[str, Any] = {
        "challenge_id": result.challenge_id,
        "submission_id": result.submission_id,
        "result_id": result.result_id,
        "state": result.state,
        "chosen_recipe": result.chosen_recipe,
        "verified_improvement": result.verified_improvement,
        "quality_floor_met": comparison.get("quality_floor_met"),
        "improved": comparison.get("improved"),
    }
    if result.brain_plan is not None:
        summary["brain_objective"] = result.brain_plan.objective
        summary["brain_strategy"] = _truncate_text(result.brain_plan.strategy, 240)
        summary["experiments"] = [
            {
                "name": experiment.name,
                "hypothesis": _truncate_text(experiment.hypothesis, 160),
                "transforms": [transform.kind for transform in experiment.transforms],
            }
            for experiment in result.brain_plan.experiments
        ]
    if isinstance(public_metrics, Mapping):
        summary["metrics"] = {
            "quality_score": public_metrics.get("quality_score"),
            "latency_us": public_metrics.get("latency_us"),
            "memory_bytes": public_metrics.get("memory_bytes"),
            "artifact_size_bytes": public_metrics.get("artifact_size_bytes"),
            "pass": public_metrics.get("pass"),
        }
    if comparison:
        summary["baseline_comparison"] = {
            "latency_improvement_pct": comparison.get("latency_improvement_pct"),
            "memory_improvement_pct": comparison.get("memory_improvement_pct"),
            "artifact_size_improvement_pct": comparison.get("artifact_size_improvement_pct"),
            "quality_delta": comparison.get("quality_delta"),
            "official_rank_score": comparison.get("official_rank_score"),
        }
    return _compact_history_entry(summary)


def _compact_history_entry(item: Mapping[str, Any]) -> dict[str, Any]:
    allowed_keys = {
        "challenge_id",
        "submission_id",
        "result_id",
        "state",
        "chosen_recipe",
        "verified_improvement",
        "quality_floor_met",
        "improved",
        "brain_objective",
        "brain_strategy",
        "experiments",
        "metrics",
        "baseline_comparison",
    }
    compact: dict[str, Any] = {}
    for key in allowed_keys:
        if key not in item:
            continue
        value = item[key]
        if isinstance(value, str):
            compact[key] = _truncate_text(value, 260)
        elif isinstance(value, (int, float, bool)) or value is None:
            compact[key] = value
        elif isinstance(value, Mapping):
            compact[key] = {
                str(child_key): child_value
                for child_key, child_value in value.items()
                if isinstance(child_value, (str, int, float, bool)) or child_value is None
            }
        elif isinstance(value, (list, tuple)):
            compact[key] = [
                child
                for child in value[:3]
                if isinstance(child, Mapping)
            ]
    return compact


def _truncate_text(value: str, max_chars: int) -> str:
    cleaned = " ".join(value.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 1].rstrip()}…"


def _resolve_candidate_bundle_dir(
    config: OrchestratorConfig,
    *,
    challenge: Mapping[str, Any] | None = None,
) -> tuple[Path, str, list[str], RecipeChoice | None, OptimizationPlan | None]:
    if config.pre_built_bundle_dir is not None:
        return config.pre_built_bundle_dir, config.selected_recipe or "pre-built", [], None, None

    if config.skip_recipe_generation:
        # operator promised the work_dir already has at least one recipe
        recipes = [get_recipe(config.selected_recipe)] if config.selected_recipe else RECIPES
        for recipe in recipes:
            candidate = config.work_dir / recipe.name
            if candidate.is_dir():
                return candidate, recipe.name, [], None, None
        raise OrchestratorError(
            f"skip_recipe_generation=True but no candidate dirs under {config.work_dir}",
        )

    if config.selected_recipe:
        selected = get_recipe(config.selected_recipe)
        results = run_candidate_recipes(
            source_model=config.source_model,
            work_dir=config.work_dir,
            recipes=[selected],
        )
        brain_pick = None
        brain_plan = None
    # When a Brain is configured, ask it for a bounded experiment plan. The
    # Brain may return a safe baseline plan on its own provider/parse errors
    # or raise in strict mode.
    elif config.brain is not None:
        brain_plan = config.brain.plan_optimization(
            challenge_spec=dict(challenge or {}),
            history=config.brain_history,
        )
        results = run_plan_experiments(
            plan=brain_plan,
            source_model=config.source_model,
            work_dir=config.work_dir,
        )
        brain_pick = (
            RecipeChoice(
                recipe_name=brain_plan.legacy_recipe_name,
                confidence=0.0,
                reasoning=brain_plan.strategy,
            )
            if brain_plan.legacy_recipe_name
            else None
        )
    else:
        brain_pick = None
        brain_plan = None
        # Preserve the historical call shape so existing test stubs that
        # patch ``run_candidate_recipes`` with a 2-kwarg signature keep
        # working without churn.
        results = run_candidate_recipes(
            source_model=config.source_model,
            work_dir=config.work_dir,
        )
    failures = [f"{r.name}: {r.error}" for r in results if not r.succeeded]
    successes = [r for r in results if r.succeeded]
    if not successes:
        raise OrchestratorError(
            "no recipe succeeded; cannot submit a candidate bundle. "
            f"Failures: {failures}",
        )
    # If the Brain picked a recipe and it succeeded, prefer that one;
    # otherwise take the first successful recipe (now reordered to match
    # the Brain's preference).
    if brain_pick is not None:
        for success in successes:
            if success.name == brain_pick.recipe_name:
                return success.output_dir, success.name, failures, brain_pick, brain_plan
    chosen = successes[0]
    return chosen.output_dir, chosen.name, failures, brain_pick, brain_plan


def _drive_uploads(
    client: CodePitClient,
    create_response: Mapping[str, Any],
    bundle_files: list[BundleFile],
) -> None:
    orchestration = create_response.get("upload_orchestration")
    if not isinstance(orchestration, Mapping):
        raise OrchestratorError("submission response did not include upload_orchestration")
    _validate_presigned_upload_ttl(orchestration)
    instructions = orchestration.get("files")
    if not isinstance(instructions, list) or not instructions:
        raise OrchestratorError("submission response did not include any upload instructions")

    by_logical_name = {file.logical_name: file for file in bundle_files}
    for instruction in instructions:
        if not isinstance(instruction, Mapping):
            raise OrchestratorError("upload instruction is not an object")
        logical_name = instruction.get("logical_name")
        upload_url = instruction.get("upload_url")
        media_type = instruction.get("media_type")
        expected_size = instruction.get("size_bytes")
        expected_sha256 = instruction.get("sha256")
        if not isinstance(logical_name, str) or not isinstance(upload_url, str):
            raise OrchestratorError(f"upload instruction is missing logical_name or upload_url: {instruction}")
        if not isinstance(media_type, str) or not media_type:
            raise OrchestratorError(f"upload instruction is missing media_type: {instruction}")
        file = by_logical_name.get(logical_name)
        if file is None:
            raise OrchestratorError(f"bundle is missing required file: {logical_name}")
        if media_type != file.media_type:
            raise OrchestratorError(
                f"upload instruction media type mismatch for {logical_name}: "
                f"engine expected {media_type}, bundle has {file.media_type}",
            )
        if isinstance(expected_size, int) and expected_size != file.size_bytes:
            raise OrchestratorError(
                f"upload instruction size mismatch for {logical_name}: "
                f"engine expected {expected_size}, bundle has {file.size_bytes}",
            )
        if isinstance(expected_sha256, str) and expected_sha256.lower() != file.sha256_hex.lower():
            raise OrchestratorError(
                f"upload instruction hash mismatch for {logical_name}",
            )
        if instruction.get("already_uploaded") is True:
            continue
        client.put_bytes(
            upload_url,
            file.content,
            content_type=media_type,
        )


def _poll_to_terminal(
    client: CodePitClient,
    submission_id: str,
    *,
    interval_s: float,
    timeout_s: float,
) -> dict:
    started = time.monotonic()
    last_state = "(none)"
    while True:
        try:
            submission = client.read_submission(submission_id)
        except ProtocolError as error:
            if error.retryable is True and (time.monotonic() - started) < timeout_s:
                time.sleep(interval_s)
                continue
            raise
        state = str(submission.get("state") or "")
        if state in TERMINAL_SUBMISSION_STATES:
            return submission
        last_state = state
        if (time.monotonic() - started) >= timeout_s:
            raise OrchestratorError(
                f"submission {submission_id} did not reach a terminal state within {timeout_s:.0f}s "
                f"(last state: {last_state})",
            )
        time.sleep(interval_s)


def _poll_public_receipt(
    client: CodePitClient,
    submission_id: str,
    *,
    interval_s: float,
    timeout_s: float,
) -> PublicReceiptObservation:
    started = time.monotonic()
    last_detail = "(none)"
    while True:
        try:
            public_submission = client.read_public_submission(submission_id)
            benchmark_state = public_submission.get("benchmark_state")
            result_id = (
                benchmark_state.get("result_id")
                if isinstance(benchmark_state, Mapping)
                else None
            )
            if isinstance(result_id, str) and result_id:
                public_result = client.read_public_result(result_id)
                baseline_comparison = public_result.get("baseline_comparison")
                comparison = (
                    dict(baseline_comparison)
                    if isinstance(baseline_comparison, Mapping)
                    else None
                )
                return PublicReceiptObservation(
                    result_id=result_id,
                    receipt_path=f"/receipts/{result_id}",
                    public_result=dict(public_result),
                    baseline_comparison=comparison,
                    verified_improvement=bool(comparison and comparison.get("improved") is True),
                )
            last_detail = str(
                public_submission.get("lifecycle_state")
                or public_submission.get("state")
                or "public projection has no result id yet",
            )
        except ProtocolError as error:
            if error.status_code != 404 and error.retryable is False:
                raise
            last_detail = f"{error.status_code}:{error.code or str(error)}"

        if (time.monotonic() - started) >= timeout_s:
            raise OrchestratorError(
                f"submission {submission_id} did not expose a public receipt within {timeout_s:.0f}s "
                f"(last public projection detail: {last_detail})",
            )
        time.sleep(interval_s)


def _read_optional_agent_summary(read: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return read()
    except ProtocolError as error:
        if error.retryable is True:
            return {}
        raise


def build_client_submission_id(
    *,
    agent_id: str,
    challenge_id: str,
    manifest_envelope: Mapping[str, Any],
) -> str:
    """Derive a stable retry id for one immutable submission intent.

    The engine's idempotency key already scopes by agent and challenge, but
    including both in the digest keeps the generated id portable in logs and
    prevents accidental collisions if a caller reuses it manually.
    """

    payload = {
        "agent_id": agent_id,
        "challenge_id": challenge_id,
        "manifest_envelope": manifest_envelope,
    }
    digest = hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()
    return f"pyopt-{digest[:32]}"


def _resolve_client_submission_id(
    explicit: str | None,
    *,
    agent_id: str,
    challenge_id: str,
    manifest_envelope: Mapping[str, Any],
) -> str:
    if explicit is None:
        return build_client_submission_id(
            agent_id=agent_id,
            challenge_id=challenge_id,
            manifest_envelope=manifest_envelope,
        )
    _validate_client_submission_id(explicit)
    return explicit


def _validate_client_submission_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise OrchestratorError("client_submission_id must be a non-empty string")
    encoded_length = len(value.encode("utf-8"))
    if encoded_length > CLIENT_SUBMISSION_ID_MAX_LENGTH:
        raise OrchestratorError(
            f"client_submission_id must be at most {CLIENT_SUBMISSION_ID_MAX_LENGTH} bytes",
        )


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _validate_presigned_upload_ttl(orchestration: Mapping[str, Any]) -> None:
    if orchestration.get("kind") != "presigned-urls":
        raise OrchestratorError("submission response did not include presigned upload URLs")
    expires_at = orchestration.get("expires_at")
    if not isinstance(expires_at, str) or not expires_at:
        raise OrchestratorError("presigned upload response did not include expires_at")
    try:
        expiry = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise OrchestratorError(f"presigned upload expires_at is invalid: {expires_at}") from error
    if expiry.tzinfo is None:
        expiry = expiry.replace(tzinfo=timezone.utc)
    ttl_seconds = (expiry.astimezone(timezone.utc) - datetime.now(timezone.utc)).total_seconds()
    if ttl_seconds <= 0:
        raise OrchestratorError("presigned upload URL has already expired")
    if ttl_seconds > MAX_UPLOAD_TTL_SECONDS + UPLOAD_TTL_CLOCK_SKEW_SECONDS:
        raise OrchestratorError(
            f"presigned upload TTL exceeds {MAX_UPLOAD_TTL_SECONDS} seconds",
        )


def _validate_selected_recipe(name: str | None) -> None:
    if name is None:
        return
    try:
        get_recipe(name)
    except ValueError as error:
        raise OrchestratorError(str(error)) from error


def _default_capabilities(declared_at_version: str) -> dict:
    return {
        "declared_artifact_lanes": ["onnx-browser-webgpu"],
        "declared_at_version": declared_at_version,
        "declared_model_classes": ["encoder-text-small"],
        "declared_runtimes": ["onnxruntime-web-webgpu"],
        "optimization_methods": ["graph-optimization", "dynamic-int8"],
    }
