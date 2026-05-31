"""CLI entrypoint for the CodePit model optimizer agent.

Three subcommands:

- ``generate`` (default when no subcommand is given): run optimization
  recipes locally and emit candidate bundles. This is what existed before
  the orchestrator landed and is preserved for back-compat.

- ``run``: drive the full production protocol against a CodePit engine —
  register or reuse the agent, discover a challenge, generate or reuse
  candidates, submit, upload, and poll until the submission reaches a
  terminal state. One-shot.

- ``run-forever``: same as ``run`` but in a supervised loop — idles when
  no challenge is available, retries transient errors, exits cleanly on
  SIGTERM/SIGINT. Use under a process supervisor (systemd, k8s, Modal)
  for the autonomous-24/7 thesis.

The legacy ``codepit-model-optimizer --work-dir <path>`` form still works
and routes to ``generate``; we detect the absence of a known subcommand
and treat the rest of argv as ``generate`` arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .brain import Brain, BrainConfig
from .brain_providers import ManagedBrainProvider
from .credential_rotation import (
    CredentialRotationConfig,
    rotate_optimizer_credentials,
)
from .brain_providers.managed import ManagedBrainProvider
from .modelbook_loop import (
    ModelbookIterationConfig,
    ModelbookIterationResult,
    run_modelbook_iteration,
    run_modelbook_loop,
)
from .orchestrator import (
    DEFAULT_REGISTER_LANE,
    DEFAULT_SOURCE_MODEL,
    DEFAULT_SOURCE_MODEL_REVISION,
    ForeverConfig,
    REGISTER_LANES,
    default_lane_runners,
    ForeverIterationOutcome,
    OrchestratorConfig,
    TinyChatRunConfig,
    register_or_load_external_agent,
    resolve_register_lane_capabilities,
    run_optimizer_agent,
    run_optimizer_agent_forever,
    run_tiny_chat_external_agent,
)
from .claim import PayoutWalletError, claim_agent_payout
from .protocol import CodePitClient
from .recipes import candidate_recipe_names, get_recipe, run_candidate_recipes, summarize_results
from .session import DEFAULT_SESSION_PATH, load_session
from .withdrawal import request_reward_withdrawal

SUBCOMMANDS = (
    "generate",
    "run",
    "run-forever",
    "rotate-credentials",
    "modelbook-run",
    "register-external",
    "tiny-chat-run",
    "claim-agent",
    "withdraw",
)
_BRAIN_PROVIDERS = ("off", "managed")
_BRAIN_TIERS = ("cheap", "mid", "premium", "network")


def main(argv: list[str] | None = None) -> None:
    args = list(argv if argv is not None else sys.argv[1:])
    subcommand = args[0] if args and args[0] in SUBCOMMANDS else "generate"
    if subcommand == args[:1]:  # pragma: no cover - defensive
        args = args[1:]
    elif args and args[0] in SUBCOMMANDS:
        args = args[1:]

    if subcommand == "run":
        _cmd_run(args)
        return

    if subcommand == "run-forever":
        _cmd_run_forever(args)
        return

    if subcommand == "rotate-credentials":
        _cmd_rotate_credentials(args)
        return

    if subcommand == "modelbook-run":
        _cmd_modelbook_run(args)
        return

    if subcommand == "register-external":
        _cmd_register_external(args)
        return

    if subcommand == "tiny-chat-run":
        _cmd_tiny_chat_run(args)
        return

    if subcommand == "claim-agent":
        _cmd_claim_agent(args)
        return

    if subcommand == "withdraw":
        _cmd_withdraw(args)
        return

    _cmd_generate(args)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def _cmd_generate(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="codepit-model-optimizer generate",
        description="Run optimization recipes locally and emit candidate bundles.",
    )
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument(
        "--recipe",
        choices=candidate_recipe_names(),
        help="Run only one known optimizer recipe.",
    )
    parsed = parser.parse_args(args)

    work_dir = Path(parsed.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    if parsed.recipe:
        results = run_candidate_recipes(
            source_model=parsed.source_model,
            work_dir=work_dir,
            recipes=[get_recipe(parsed.recipe)],
        )
    else:
        results = run_candidate_recipes(source_model=parsed.source_model, work_dir=work_dir)

    print(f"{summarize_results(results)} in {work_dir}")
    for result in results:
        state = "ok" if result.succeeded else "failed"
        line = f"- {result.name}: {state} ({result.output_dir})"
        if result.error:
            line = f"{line}: {result.error}"
        print(line)
    if not any(result.succeeded for result in results):
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# run (orchestrator)
# ---------------------------------------------------------------------------


def _add_brain_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--brain-provider",
        choices=_BRAIN_PROVIDERS,
        default=os.environ.get("CODEPIT_OPTIMIZER_BRAIN_PROVIDER", "off"),
        help=(
            "Strategic recipe picker. 'managed' calls the engine brain gateway "
            "with this agent's runtime credential; 'off' preserves local-only behavior."
        ),
    )
    parser.add_argument(
        "--brain-tier",
        choices=_BRAIN_TIERS,
        default=os.environ.get("CODEPIT_OPTIMIZER_BRAIN_TIER", "premium"),
        help="Managed brain tier to request from the engine. Default: premium.",
    )
    parser.add_argument(
        "--brain-timeout-seconds",
        type=float,
        default=_env_float("CODEPIT_OPTIMIZER_BRAIN_TIMEOUT_SECONDS", 60.0),
    )
    parser.add_argument(
        "--require-llm-brain",
        action="store_true",
        default=_env_bool("CODEPIT_REQUIRE_LLM_BRAIN", False),
        help=(
            "Fail instead of falling back to baseline-export when the configured "
            "brain provider is unavailable or returns invalid JSON."
        ),
    )


def _build_brain(
    *,
    provider_name: str,
    base_url: str,
    runtime_credential: str | None,
    tier: str,
    timeout_s: float,
    require_llm_brain: bool,
) -> Brain | None:
    if provider_name == "off":
        if require_llm_brain:
            raise SystemExit("--require-llm-brain needs --brain-provider managed")
        return None
    if provider_name != "managed":
        raise SystemExit(f"Unsupported brain provider: {provider_name}")
    if not runtime_credential:
        raise SystemExit("--brain-provider managed needs --runtime-credential")
    provider = ManagedBrainProvider(
        base_url=base_url,
        bearer_token=runtime_credential,
        timeout_s=timeout_s,
    )
    return Brain(
        config=BrainConfig(
            provider_name="managed",
            tier=tier,
            fallback_on_error=not require_llm_brain,
            action_id_prefix=f"pyopt-{int(time.time() * 1000)}-{os.getpid()}",
        ),
        provider=provider,
    )


def _env_bool(name: str, fallback: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return fallback
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, fallback: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return fallback
    try:
        parsed = float(value)
    except ValueError:
        return fallback
    return parsed if parsed > 0 else fallback


def _cmd_run(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="codepit-model-optimizer run",
        description="Register, optimize, submit, and poll a CodePit V2 challenge end-to-end.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CODEPIT_V2_BASE_URL", "http://127.0.0.1:3004"),
        help="Engine base URL. Default: $CODEPIT_V2_BASE_URL or http://127.0.0.1:3004",
    )
    parser.add_argument("--work-dir", required=True, help="Where candidate bundles are emitted/read.")
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-model-revision", default=DEFAULT_SOURCE_MODEL_REVISION)
    parser.add_argument(
        "--recipe",
        choices=candidate_recipe_names(),
        help="Run and submit only one known optimizer recipe.",
    )
    parser.add_argument(
        "--challenge-id",
        default=os.environ.get("CODEPIT_V2_CHALLENGE_ID"),
        help="Pin to a specific challenge instead of using /v1/challenges/next.",
    )
    parser.add_argument(
        "--client-submission-id",
        default=os.environ.get("CODEPIT_V2_CLIENT_SUBMISSION_ID"),
        help=(
            "Optional retry key for this exact submission intent. "
            "Defaults to a deterministic hash of the manifest intent."
        ),
    )
    parser.add_argument(
        "--display-name",
        default=os.environ.get("CODEPIT_V2_AGENT_DISPLAY_NAME"),
    )
    parser.add_argument(
        "--private-key",
        default=os.environ.get("CODEPIT_V2_AGENT_PRIVATE_KEY"),
        help="Optional 0x-prefixed signer private key. Defaults to ephemeral.",
    )
    parser.add_argument(
        "--agent-wallet-private-key",
        default=os.environ.get("CODEPIT_V2_AGENT_WALLET_PRIVATE_KEY"),
        help=(
            "Optional 0x-prefixed agent wallet private key. Defaults to the "
            "persisted session wallet or a new local wallet."
        ),
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("CODEPIT_V2_AGENT_ID"),
        help="Use an already-provisioned agent id instead of registering.",
    )
    parser.add_argument(
        "--runtime-credential",
        default=os.environ.get("CODEPIT_V2_RUNTIME_CREDENTIAL"),
        help="Bearer credential for --agent-id / managed-runtime sessions.",
    )
    parser.add_argument(
        "--runtime-credential-id",
        default=os.environ.get("CODEPIT_V2_RUNTIME_CREDENTIAL_ID"),
    )
    parser.add_argument(
        "--session-path",
        default=os.environ.get("CODEPIT_V2_SESSION_PATH"),
        help=f"Where to persist the agent session. Default: {DEFAULT_SESSION_PATH}",
    )
    parser.add_argument("--no-session-persist", action="store_true", help="Do not persist the agent session.")
    parser.add_argument(
        "--pre-built-bundle-dir",
        help="Use an existing bundle directory instead of running recipes.",
    )
    parser.add_argument(
        "--skip-recipe-generation",
        action="store_true",
        help="Reuse candidate dirs already under --work-dir; do not re-run recipes.",
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=3.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=5 * 60.0)
    parser.add_argument(
        "--receipt-poll-timeout-seconds",
        type=float,
        default=None,
        help="How long to wait for the public receipt projection after verification. Defaults to --poll-timeout-seconds.",
    )
    _add_brain_args(parser)
    parsed = parser.parse_args(args)

    session_path: Path | None
    if parsed.no_session_persist:
        session_path = None
    elif parsed.session_path:
        session_path = Path(parsed.session_path)
    else:
        session_path = DEFAULT_SESSION_PATH

    config = OrchestratorConfig(
        base_url=parsed.base_url,
        work_dir=Path(parsed.work_dir),
        source_model=parsed.source_model,
        source_model_revision=parsed.source_model_revision,
        private_key=parsed.private_key,
        agent_wallet_private_key=parsed.agent_wallet_private_key,
        agent_id=parsed.agent_id,
        runtime_credential=parsed.runtime_credential,
        runtime_credential_id=parsed.runtime_credential_id,
        challenge_id=parsed.challenge_id,
        client_submission_id=parsed.client_submission_id,
        display_name=parsed.display_name,
        poll_interval_s=parsed.poll_interval_seconds,
        poll_timeout_s=parsed.poll_timeout_seconds,
        receipt_poll_timeout_s=parsed.receipt_poll_timeout_seconds,
        session_path=session_path,
        pre_built_bundle_dir=Path(parsed.pre_built_bundle_dir) if parsed.pre_built_bundle_dir else None,
        skip_recipe_generation=parsed.skip_recipe_generation,
        selected_recipe=parsed.recipe,
        brain=_build_brain(
            provider_name=parsed.brain_provider,
            base_url=parsed.base_url,
            runtime_credential=parsed.runtime_credential,
            tier=parsed.brain_tier,
            timeout_s=parsed.brain_timeout_seconds,
            require_llm_brain=parsed.require_llm_brain,
        ),
    )

    result = run_optimizer_agent(config)

    payload = asdict(result)
    payload["bundle_dir"] = str(result.bundle_dir)
    print(json.dumps(payload, indent=2, default=str))

    if result.state not in {"VERIFIED", "SETTLED", "PUBLISHED"}:
        # Submission ran to a terminal state, but not a happy one. Surface
        # this as a non-zero exit so CI/operators notice.
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# run-forever (autonomous 24/7 supervisor)
# ---------------------------------------------------------------------------


def _cmd_run_forever(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="codepit-model-optimizer run-forever",
        description=(
            "Run the optimizer agent in a supervised loop. Idles when no "
            "challenge is available, retries transient errors, exits "
            "cleanly on SIGTERM/SIGINT."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CODEPIT_V2_BASE_URL", "http://127.0.0.1:3004"),
    )
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--source-model", default=DEFAULT_SOURCE_MODEL)
    parser.add_argument("--source-model-revision", default=DEFAULT_SOURCE_MODEL_REVISION)
    parser.add_argument(
        "--recipe",
        choices=candidate_recipe_names(),
        help="Run and submit only one known optimizer recipe per iteration.",
    )
    parser.add_argument(
        "--display-name",
        default=os.environ.get("CODEPIT_V2_AGENT_DISPLAY_NAME"),
    )
    parser.add_argument(
        "--private-key",
        default=os.environ.get("CODEPIT_V2_AGENT_PRIVATE_KEY"),
    )
    parser.add_argument(
        "--agent-wallet-private-key",
        default=os.environ.get("CODEPIT_V2_AGENT_WALLET_PRIVATE_KEY"),
        help="Optional 0x-prefixed agent wallet private key.",
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("CODEPIT_V2_AGENT_ID"),
        help="Use an already-provisioned agent id instead of registering.",
    )
    parser.add_argument(
        "--runtime-credential",
        default=os.environ.get("CODEPIT_V2_RUNTIME_CREDENTIAL"),
        help="Bearer credential for --agent-id / managed-runtime sessions.",
    )
    parser.add_argument(
        "--runtime-credential-id",
        default=os.environ.get("CODEPIT_V2_RUNTIME_CREDENTIAL_ID"),
    )
    parser.add_argument(
        "--session-path",
        default=os.environ.get("CODEPIT_V2_SESSION_PATH"),
        help=f"Where to persist the agent session. Default: {DEFAULT_SESSION_PATH}",
    )
    parser.add_argument("--no-session-persist", action="store_true")
    parser.add_argument(
        "--pre-built-bundle-dir",
        help="Use an existing bundle directory for every iteration.",
    )
    parser.add_argument(
        "--skip-recipe-generation",
        action="store_true",
        help="Reuse candidate dirs already under --work-dir; do not re-run recipes.",
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=3.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=5 * 60.0)
    parser.add_argument(
        "--receipt-poll-timeout-seconds",
        type=float,
        default=None,
        help="How long each iteration waits for the public receipt projection after verification.",
    )
    parser.add_argument(
        "--idle-sleep-seconds",
        type=float,
        default=30.0,
        help="Sleep when /v1/challenges/next reports nothing eligible.",
    )
    parser.add_argument(
        "--error-backoff-seconds",
        type=float,
        default=15.0,
        help="Sleep after a transient protocol error before retrying.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Stop after N completed iterations (any kind). Default: unlimited.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=None,
        help="Stop after this many wall-clock seconds. Default: unlimited.",
    )
    parser.add_argument(
        "--reuse-work-dir",
        action="store_true",
        help="Disable per-iteration work subdirectories. Default: each iteration uses --work-dir/iter-NNNNNN.",
    )
    _add_brain_args(parser)
    parsed = parser.parse_args(args)

    session_path: Path | None
    if parsed.no_session_persist:
        session_path = None
    elif parsed.session_path:
        session_path = Path(parsed.session_path)
    else:
        session_path = DEFAULT_SESSION_PATH

    base_config = OrchestratorConfig(
        base_url=parsed.base_url,
        work_dir=Path(parsed.work_dir),
        source_model=parsed.source_model,
        source_model_revision=parsed.source_model_revision,
        private_key=parsed.private_key,
        agent_wallet_private_key=parsed.agent_wallet_private_key,
        agent_id=parsed.agent_id,
        runtime_credential=parsed.runtime_credential,
        runtime_credential_id=parsed.runtime_credential_id,
        challenge_id=None,
        display_name=parsed.display_name,
        poll_interval_s=parsed.poll_interval_seconds,
        poll_timeout_s=parsed.poll_timeout_seconds,
        receipt_poll_timeout_s=parsed.receipt_poll_timeout_seconds,
        session_path=session_path,
        pre_built_bundle_dir=Path(parsed.pre_built_bundle_dir) if parsed.pre_built_bundle_dir else None,
        skip_recipe_generation=parsed.skip_recipe_generation,
        selected_recipe=parsed.recipe,
        brain=_build_brain(
            provider_name=parsed.brain_provider,
            base_url=parsed.base_url,
            runtime_credential=parsed.runtime_credential,
            tier=parsed.brain_tier,
            timeout_s=parsed.brain_timeout_seconds,
            require_llm_brain=parsed.require_llm_brain,
        ),
    )

    forever = ForeverConfig(
        base_config=base_config,
        idle_sleep_s=parsed.idle_sleep_seconds,
        error_backoff_s=parsed.error_backoff_seconds,
        max_iterations=parsed.max_iterations,
        max_runtime_s=parsed.max_runtime_seconds,
        fresh_work_dir_per_iteration=not parsed.reuse_work_dir,
        on_iteration=_print_iteration_outcome,
        # Lane-runner dispatch (#87): a `run-forever` invocation can claim
        # any eligible challenge on any lane in the default registry. Power
        # users can monkeypatch / wrap the kit to register additional lanes
        # without modifying the supervisor.
        lane_runners=default_lane_runners(),
    )

    summary = run_optimizer_agent_forever(forever)

    print(
        json.dumps(
            {
                "stopped_reason": summary.stopped_reason,
                "iterations_started": summary.iterations_started,
                "iterations_completed": summary.iterations_completed,
                "transient_error_count": summary.transient_error_count,
                "terminal_state_counts": summary.terminal_state_counts,
                "last_error": summary.last_error,
            },
            indent=2,
            default=str,
        ),
    )

    if summary.stopped_reason == "fatal_error":
        raise SystemExit(2)


# ---------------------------------------------------------------------------
# rotate-credentials
# ---------------------------------------------------------------------------


def _cmd_rotate_credentials(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="codepit-model-optimizer rotate-credentials",
        description="Rotate an agent runtime credential using a signer-bound intent.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CODEPIT_V2_BASE_URL", "http://127.0.0.1:3004"),
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("CODEPIT_V2_AGENT_ID"),
        help="Agent id to rotate. Defaults to the persisted session agent.",
    )
    parser.add_argument(
        "--private-key",
        default=os.environ.get("CODEPIT_V2_AGENT_PRIVATE_KEY"),
        help="0x-prefixed signer private key. Defaults to the persisted session signer.",
    )
    parser.add_argument(
        "--session-path",
        default=os.environ.get("CODEPIT_V2_SESSION_PATH"),
        help=f"Where to persist the fresh credential. Default: {DEFAULT_SESSION_PATH}",
    )
    parser.add_argument("--no-session-persist", action="store_true")
    parsed = parser.parse_args(args)

    if parsed.no_session_persist:
        session_path = None
    elif parsed.session_path:
        session_path = Path(parsed.session_path)
    else:
        session_path = DEFAULT_SESSION_PATH

    result = rotate_optimizer_credentials(
        CredentialRotationConfig(
            base_url=parsed.base_url,
            agent_id=parsed.agent_id,
            private_key=parsed.private_key,
            session_path=session_path,
        )
    )

    print(json.dumps(asdict(result), indent=2))


def _print_iteration_outcome(iteration: int, outcome: ForeverIterationOutcome) -> None:
    payload: dict = {
        "iteration": iteration,
        "kind": outcome.kind,
        "duration_s": round(outcome.finished_at - outcome.started_at, 3),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if outcome.kind == "result" and outcome.result is not None:
        payload["agent_id"] = outcome.result.agent_id
        payload["submission_id"] = outcome.result.submission_id
        payload["client_submission_id"] = outcome.result.client_submission_id
        payload["result_id"] = outcome.result.result_id
        payload["receipt_path"] = outcome.result.receipt_path
        payload["verified_improvement"] = outcome.result.verified_improvement
        payload["state"] = outcome.result.state
        payload["proof_record_id"] = outcome.result.proof_record_id
        payload["chosen_recipe"] = outcome.result.chosen_recipe
        if outcome.result.recipe_failures:
            payload["recipe_failures"] = outcome.result.recipe_failures
    if outcome.error_message:
        payload["error_message"] = outcome.error_message
    if outcome.error_code:
        payload["error_code"] = outcome.error_code
    print(json.dumps(payload, default=str), flush=True)


# ---------------------------------------------------------------------------
# modelbook-run (V2 SML Modelbook iteration)
# ---------------------------------------------------------------------------


def _cmd_modelbook_run(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="codepit-model-optimizer modelbook-run",
        description=(
            "Run one or more Modelbook iterations against a CodePit V2 engine. "
            "Each iteration discovers an available Modelbook, reads its context, "
            "creates a TrainingRun, records decisions + events, performs the "
            "deterministic Tiny Chat training fixture, and registers checksum-backed "
            "artifact files."
        ),
    )
    parser.add_argument(
        "--engine-url",
        default=os.environ.get("CODEPIT_ENGINE_URL"),
        help="Base URL of the CodePit V2 engine (env CODEPIT_ENGINE_URL).",
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("CODEPIT_AGENT_ID"),
        help="Registered agent id (env CODEPIT_AGENT_ID).",
    )
    parser.add_argument(
        "--credential",
        default=os.environ.get("CODEPIT_RUNTIME_CREDENTIAL"),
        help="Bearer runtime credential (env CODEPIT_RUNTIME_CREDENTIAL).",
    )
    parser.add_argument(
        "--modelbook-id",
        default=None,
        help="Pin the iteration to a specific Modelbook id. Default: first available.",
    )
    parser.add_argument(
        "--recipe-kind",
        default=None,
        help="Pin the recipe. Must be in policy.allowed_training_methods.",
    )
    parser.add_argument(
        "--artifact-output-dir",
        default=os.environ.get("CODEPIT_MODELBOOK_ARTIFACT_DIR"),
        help=(
            "Directory for local Tiny Chat artifacts. Default: "
            ".local/modelbook-artifacts."
        ),
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        default=_env_bool("CODEPIT_MODELBOOK_SUBMIT", False),
        help=(
            "Create a canonical /v1/submissions record, upload the Tiny Chat "
            "bundle, and attach the submission to the TrainingRun."
        ),
    )
    parser.add_argument(
        "--challenge-id",
        default=os.environ.get("CODEPIT_MODELBOOK_CHALLENGE_ID"),
        help="Verifier challenge id for --submit. Default: /v1/challenges/next.",
    )
    parser.add_argument(
        "--client-submission-id",
        default=os.environ.get("CODEPIT_MODELBOOK_CLIENT_SUBMISSION_ID"),
        help="Optional idempotency key for the canonical submission created by --submit.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1,
        help=(
            "Stop after this many non-skipped iterations. Use 0 to loop until "
            "interrupted. Default 1."
        ),
    )
    parser.add_argument(
        "--idle-sleep-seconds",
        type=float,
        default=5.0,
        help="Seconds to sleep when no Modelbook is available. Default 5.0.",
    )
    parser.add_argument(
        "--no-brain",
        action="store_true",
        help=(
            "Skip the LLM brain and use deterministic heuristics for every "
            "decision. Default behavior calls the engine /v2/brain/generate "
            "endpoint and records the real provider+model on each decision."
        ),
    )
    parser.add_argument(
        "--brain-tier",
        choices=("cheap", "mid", "premium", "network"),
        default=os.environ.get("CODEPIT_BRAIN_TIER", "cheap"),
        help=(
            "Brain tier sent to /v2/brain/generate. Resolved on the engine "
            "to a real provider+model. Default 'cheap' (Llama 3.1 8B via Groq)."
        ),
    )
    parsed = parser.parse_args(args)

    if not parsed.engine_url:
        parser.error("missing required argument: --engine-url (or env CODEPIT_ENGINE_URL)")

    agent_id = parsed.agent_id
    credential = parsed.credential
    if not agent_id or not credential:
        session = load_session(DEFAULT_SESSION_PATH)
        if session is not None and session.base_url.rstrip("/") == parsed.engine_url.rstrip("/"):
            agent_id = agent_id or session.agent_id
            credential = credential or session.runtime_credential
            print(
                f"# using persisted session at {DEFAULT_SESSION_PATH} "
                f"(agent_id={agent_id})",
                file=sys.stderr,
            )

    if not agent_id or not credential:
        parser.error(
            "no agent credentials available — pass --agent-id + --credential, "
            "set CODEPIT_AGENT_ID + CODEPIT_RUNTIME_CREDENTIAL, or run "
            "`register-external` first to provision a fresh external agent."
        )

    client = CodePitClient(
        parsed.engine_url,
        agent_id=agent_id,
        credential=credential,
    )
    brain = (
        None
        if parsed.no_brain
        else ManagedBrainProvider(
            base_url=parsed.engine_url,
            bearer_token=credential,
        )
    )
    iteration_config = ModelbookIterationConfig(
        modelbook_id=parsed.modelbook_id,
        recipe_kind=parsed.recipe_kind,
        artifact_output_dir=parsed.artifact_output_dir,
        brain=brain,
        brain_tier=parsed.brain_tier,
        submit=parsed.submit,
        challenge_id=parsed.challenge_id,
        client_submission_id=parsed.client_submission_id,
    )

    if parsed.max_iterations == 1:
        result = run_modelbook_iteration(client, iteration_config)
        _print_iteration_result(result)
        if result.skipped_reason == "no_available_modelbook":
            raise SystemExit(2)
        return

    results = run_modelbook_loop(
        client,
        iteration_config,
        max_iterations=parsed.max_iterations if parsed.max_iterations > 0 else None,
        idle_sleep_seconds=parsed.idle_sleep_seconds,
    )
    for result in results:
        _print_iteration_result(result)


def _print_iteration_result(result: ModelbookIterationResult) -> None:
    payload = {
        "modelbook_id": result.modelbook_id,
        "training_run_id": result.training_run_id,
        "recipe_kind": result.recipe_kind,
        "decisions_recorded": result.decisions_recorded,
        "events_emitted": result.events_emitted,
        "artifact_set_id": result.artifact_set_id,
        "challenge_id": result.challenge_id,
        "submission_id": result.submission_id,
        "submission_state": result.submission_state,
        "skipped_reason": result.skipped_reason,
        "stub_training_used": result.stub_training_used,
        "brain_driven": result.brain_driven,
        "brain_provider": result.brain_provider,
        "brain_model": result.brain_model,
        "notes": result.notes,
    }
    print(json.dumps(payload, default=str), flush=True)


# ---------------------------------------------------------------------------
# register-external (provision a fresh external agent without running the loop)
# ---------------------------------------------------------------------------


def _cmd_register_external(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="codepit-model-optimizer register-external",
        description=(
            "Register a fresh external agent on the CodePit V2 engine. The "
            "signer key + runtime credential are persisted to "
            "~/.codepit/agent.json (mode 0600). After this, `modelbook-run` "
            "auto-uses these credentials without further plumbing."
        ),
    )
    parser.add_argument(
        "--engine-url",
        default=os.environ.get("CODEPIT_ENGINE_URL"),
        help="Base URL of the CodePit V2 engine (env CODEPIT_ENGINE_URL).",
    )
    parser.add_argument(
        "--display-name",
        default=os.environ.get("CODEPIT_V2_AGENT_DISPLAY_NAME"),
        help="Optional human-readable display name for the new agent.",
    )
    parser.add_argument(
        "--session-path",
        type=Path,
        default=DEFAULT_SESSION_PATH,
        help=f"Path to the session file. Default: {DEFAULT_SESSION_PATH}",
    )
    parser.add_argument(
        "--lane",
        choices=REGISTER_LANES,
        default=DEFAULT_REGISTER_LANE,
        help=(
            "Capability lane to declare at registration. Default 'tiny-chat' "
            "(ollama-gguf-local / chat-causal-small) matches the live challenge "
            "so `modelbook-run --submit` works without extra flags. Capabilities "
            "cannot be changed after registration."
        ),
    )
    parsed = parser.parse_args(args)

    if not parsed.engine_url:
        parser.error("missing required argument: --engine-url (or env CODEPIT_ENGINE_URL)")

    capabilities = resolve_register_lane_capabilities(parsed.lane)
    session, reused = register_or_load_external_agent(
        base_url=parsed.engine_url,
        session_path=parsed.session_path,
        display_name=parsed.display_name,
        capabilities=capabilities,
    )

    # IMPORTANT: never print the runtime credential or signer private key.
    # Only the agent_id + signer_address + path are safe for stdout.
    payload = {
        "status": "reused" if reused else "registered",
        "agent_id": session.agent_id,
        "signer_address": session.signer_address,
        "session_path": str(parsed.session_path),
        "engine_url": session.base_url,
        "lane": parsed.lane,
        "declared_model_classes": capabilities["declared_model_classes"],
        "declared_artifact_lanes": capabilities["declared_artifact_lanes"],
        "next_step": (
            "python -m codepit_optimizer.cli modelbook-run --submit --max-iterations 1"
        ),
    }
    print(json.dumps(payload, default=str, indent=2), flush=True)


def _cmd_tiny_chat_run(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="codepit-model-optimizer tiny-chat-run",
        description=(
            "Register (or reuse), build a real GGUF on the agent's own compute, "
            "assemble the ollama-gguf-local bundle, submit over the canonical "
            "protocol, and poll until the submission reaches a terminal state."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CODEPIT_V2_BASE_URL", "http://127.0.0.1:3004"),
        help="Engine base URL. Default: $CODEPIT_V2_BASE_URL or http://127.0.0.1:3004",
    )
    parser.add_argument("--work-dir", required=True, help="Where the GGUF is built and staged.")
    parser.add_argument(
        "--challenge-id",
        default=os.environ.get("CODEPIT_V2_CHALLENGE_ID"),
        help="Pin to a specific ollama-gguf-local challenge instead of /v1/challenges/next.",
    )
    parser.add_argument(
        "--target",
        choices=["auto", "sponsor", "bootstrap"],
        default=os.environ.get("CODEPIT_V2_CHALLENGE_TARGET", "auto"),
        help=(
            "Challenge targeting when --challenge-id is unset. 'sponsor' enters the "
            "richest eligible open sponsor competition (a rewarded challenge); "
            "'auto'/'bootstrap' use /v1/challenges/next. Default: auto."
        ),
    )
    parser.add_argument(
        "--base-model-ref",
        default=os.environ.get("CODEPIT_TINY_CHAT_BASE_MODEL_REF", "hf://Qwen/Qwen2.5-0.5B-Instruct"),
        help="Source model the agent optimizes. Default: Qwen2.5-0.5B-Instruct.",
    )
    parser.add_argument(
        "--quantization-profile",
        default=os.environ.get("CODEPIT_TINY_CHAT_QUANT_PROFILE", "q4_k_m"),
        help="Quantization profile for the GGUF build. Default: q4_k_m.",
    )
    parser.add_argument(
        "--optimization-method",
        action="append",
        dest="optimization_methods",
        help="Optimization method to declare in the manifest (repeatable). Defaults to the quant profile.",
    )
    parser.add_argument(
        "--gguf-path",
        default=os.environ.get("CODEPIT_TINY_CHAT_GGUF_PATH"),
        help=(
            "Submit this already-built real GGUF as-is (the agent built it with its "
            "own toolchain) instead of building one via the env-gated llama.cpp pipeline."
        ),
    )
    parser.add_argument(
        "--client-submission-id",
        default=os.environ.get("CODEPIT_V2_CLIENT_SUBMISSION_ID"),
        help="Optional retry key. Defaults to a deterministic hash of the manifest intent.",
    )
    parser.add_argument(
        "--display-name",
        default=os.environ.get("CODEPIT_V2_AGENT_DISPLAY_NAME"),
    )
    parser.add_argument(
        "--private-key",
        default=os.environ.get("CODEPIT_V2_AGENT_PRIVATE_KEY"),
        help="Optional 0x-prefixed signer private key. Defaults to ephemeral/persisted.",
    )
    parser.add_argument(
        "--agent-wallet-private-key",
        default=os.environ.get("CODEPIT_V2_AGENT_WALLET_PRIVATE_KEY"),
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("CODEPIT_V2_AGENT_ID"),
        help="Use an already-provisioned agent id instead of registering.",
    )
    parser.add_argument(
        "--runtime-credential",
        default=os.environ.get("CODEPIT_V2_RUNTIME_CREDENTIAL"),
        help="Bearer credential for --agent-id / managed-runtime sessions.",
    )
    parser.add_argument(
        "--runtime-credential-id",
        default=os.environ.get("CODEPIT_V2_RUNTIME_CREDENTIAL_ID"),
    )
    parser.add_argument(
        "--session-path",
        default=os.environ.get("CODEPIT_V2_SESSION_PATH"),
        help=f"Where to persist the agent session. Default: {DEFAULT_SESSION_PATH}",
    )
    parser.add_argument("--no-session-persist", action="store_true", help="Do not persist the agent session.")
    parser.add_argument(
        "--allow-unbound-payout",
        action="store_true",
        help=(
            "Proceed with a rewarded --target sponsor run even when no payout "
            "address is bound. By default the run is refused so a verified reward "
            "is not silently forfeited; bind a wallet via 'claim-agent' instead."
        ),
    )
    parser.add_argument("--poll-interval-seconds", type=float, default=3.0)
    parser.add_argument("--poll-timeout-seconds", type=float, default=5 * 60.0)
    parser.add_argument(
        "--receipt-poll-timeout-seconds",
        type=float,
        default=None,
        help="How long to wait for the public receipt after verification. Defaults to --poll-timeout-seconds.",
    )
    parsed = parser.parse_args(args)

    session_path: Path | None
    if parsed.no_session_persist:
        session_path = None
    elif parsed.session_path:
        session_path = Path(parsed.session_path)
    else:
        session_path = DEFAULT_SESSION_PATH

    config = TinyChatRunConfig(
        base_url=parsed.base_url,
        work_dir=Path(parsed.work_dir),
        challenge_id=parsed.challenge_id,
        target=parsed.target,
        allow_unbound_payout=parsed.allow_unbound_payout,
        base_model_ref=parsed.base_model_ref,
        quantization_profile=parsed.quantization_profile,
        optimization_methods=parsed.optimization_methods,
        gguf_path=Path(parsed.gguf_path) if parsed.gguf_path else None,
        private_key=parsed.private_key,
        agent_wallet_private_key=parsed.agent_wallet_private_key,
        agent_id=parsed.agent_id,
        runtime_credential=parsed.runtime_credential,
        runtime_credential_id=parsed.runtime_credential_id,
        display_name=parsed.display_name,
        client_submission_id=parsed.client_submission_id,
        poll_interval_s=parsed.poll_interval_seconds,
        poll_timeout_s=parsed.poll_timeout_seconds,
        receipt_poll_timeout_s=parsed.receipt_poll_timeout_seconds,
        session_path=session_path,
    )

    result = run_tiny_chat_external_agent(config)

    payload = asdict(result)
    payload["bundle_dir"] = str(result.bundle_dir)
    print(json.dumps(payload, indent=2, default=str))

    if result.state not in {"VERIFIED", "SETTLED", "PUBLISHED"}:
        raise SystemExit(2)


def _cmd_claim_agent(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="codepit-model-optimizer claim-agent",
        description=(
            "Bind an agent's payout address via the owner-claim flow so verified "
            "sponsor rewards are not forfeited. The OWNER signs the canonical claim "
            "message with the wallet they control; that wallet becomes the payout "
            "address and the only address authorized to withdraw. Reuses the "
            "existing POST /v1/agents/:id/claim endpoint — no parallel claim path."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CODEPIT_V2_BASE_URL", "http://127.0.0.1:3004"),
        help="Engine base URL. Default: $CODEPIT_V2_BASE_URL or http://127.0.0.1:3004",
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("CODEPIT_V2_AGENT_ID"),
        help="Agent to claim. Falls back to the persisted session's agent_id.",
    )
    parser.add_argument(
        "--agent-signer-address",
        default=os.environ.get("CODEPIT_V2_AGENT_SIGNER_ADDRESS"),
        help="The agent's signer address (binds the claim). Falls back to the session.",
    )
    parser.add_argument(
        "--claim-token",
        default=os.environ.get("CODEPIT_V2_CLAIM_TOKEN"),
        help="Single-use claim token printed at registration (claim.claim_token).",
    )
    parser.add_argument(
        "--owner-claim-private-key",
        default=os.environ.get("CODEPIT_V2_OWNER_CLAIM_PRIVATE_KEY"),
        help=(
            "0x-prefixed private key of the OWNER wallet that will receive payouts. "
            "Prefer the env var over the flag. The wallet address is bound as the "
            "payout address."
        ),
    )
    parser.add_argument(
        "--i-control-the-payout-wallet",
        dest="payout_wallet_ack",
        action="store_true",
        default=_env_bool("CODEPIT_V2_PAYOUT_WALLET_ACK", False),
        help=(
            "Acknowledge that you personally control AND have backed up the owner "
            "wallet's private key. Rewards are paid to that address and only its "
            "key can move them — a lost key means permanently locked funds. "
            "Required (or set CODEPIT_V2_PAYOUT_WALLET_ACK=1)."
        ),
    )
    parser.add_argument(
        "--session-path",
        default=os.environ.get("CODEPIT_V2_SESSION_PATH"),
        help=f"Session to read agent_id / signer from. Default: {DEFAULT_SESSION_PATH}",
    )
    parsed = parser.parse_args(args)

    session = None
    session_path = Path(parsed.session_path) if parsed.session_path else DEFAULT_SESSION_PATH
    loaded = load_session(session_path)
    if loaded is not None and loaded.base_url == parsed.base_url:
        session = loaded

    agent_id = parsed.agent_id or (session.agent_id if session is not None else None)
    agent_signer_address = parsed.agent_signer_address or (
        session.signer_address if session is not None else None
    )

    missing = [
        name
        for name, value in (
            ("--agent-id", agent_id),
            ("--agent-signer-address", agent_signer_address),
            ("--claim-token", parsed.claim_token),
            ("--owner-claim-private-key", parsed.owner_claim_private_key),
        )
        if not value
    ]
    if missing:
        parser.error(f"missing required input(s): {', '.join(missing)}")

    if not parsed.payout_wallet_ack:
        parser.error(
            "refusing to bind a payout wallet without acknowledgment. Rewards are "
            "paid to the owner wallet and only its private key can move them — if "
            "you lose that key the funds are locked forever. Use a wallet you "
            "personally control and have backed up, then re-run with "
            "--i-control-the-payout-wallet (or set CODEPIT_V2_PAYOUT_WALLET_ACK=1)."
        )

    # Footgun guard: never bind the agent's own auto-generated signer/wallet as
    # the payout address. Those ephemeral keys live only in the local session
    # and are rarely backed up — binding one risks locking rewards and lets the
    # autonomous agent move funds.
    forbidden_payout_addresses = [
        addr
        for addr in (
            agent_signer_address,
            session.agent_wallet_address if session is not None else None,
            session.signer_address if session is not None else None,
        )
        if addr
    ]

    client = CodePitClient(parsed.base_url)
    try:
        response = claim_agent_payout(
            client,
            agent_id=agent_id,
            agent_signer_address=agent_signer_address,
            claim_token=parsed.claim_token,
            owner_private_key=parsed.owner_claim_private_key,
            forbidden_payout_addresses=forbidden_payout_addresses,
        )
    except PayoutWalletError as exc:
        parser.error(str(exc))

    print(json.dumps(response, indent=2, default=str))

    if response.get("claim_status") != "claimed":
        raise SystemExit(2)

    payout_address = response.get("payout_address") or "(see response above)"
    print(
        "\n"
        "──────────────────────────────────────────────────────────────\n"
        " PAYOUT WALLET BOUND — these rewards are YOURS\n"
        "──────────────────────────────────────────────────────────────\n"
        f"  Rewards are paid to: {payout_address}\n"
        "  Only this wallet's private key can move the funds. CodePit never\n"
        "  holds your key. Back it up now — a lost key means locked rewards.\n"
        "──────────────────────────────────────────────────────────────",
        file=sys.stderr,
    )


def _cmd_withdraw(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="codepit-model-optimizer withdraw",
        description=(
            "Withdraw a settled reward. The OWNER signs the canonical withdrawal "
            "message with the wallet bound as the agent's payout address (at claim "
            "time) and posts it to POST /v1/agents/:id/withdrawals. The wallet "
            "signature is the auth; the runtime credential cannot move funds."
        ),
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("CODEPIT_V2_BASE_URL", "http://127.0.0.1:3004"),
        help="Engine base URL. Default: $CODEPIT_V2_BASE_URL or http://127.0.0.1:3004",
    )
    parser.add_argument(
        "--agent-id",
        default=os.environ.get("CODEPIT_V2_AGENT_ID"),
        help="Agent whose reward is being withdrawn. Falls back to the session.",
    )
    parser.add_argument(
        "--amount-raw",
        required=True,
        help="Amount to withdraw in raw token units (positive decimal integer).",
    )
    parser.add_argument(
        "--client-withdrawal-id",
        required=True,
        help="Idempotency key for this withdrawal intent (1-128 of [A-Za-z0-9._:-]).",
    )
    parser.add_argument(
        "--owner-withdraw-private-key",
        default=os.environ.get("CODEPIT_V2_OWNER_WITHDRAW_PRIVATE_KEY"),
        help=(
            "0x-prefixed private key of the OWNER wallet (the bound payout address). "
            "Prefer the env var over the flag."
        ),
    )
    parser.add_argument(
        "--session-path",
        default=os.environ.get("CODEPIT_V2_SESSION_PATH"),
        help=f"Session to read agent_id from. Default: {DEFAULT_SESSION_PATH}",
    )
    parsed = parser.parse_args(args)

    session_path = Path(parsed.session_path) if parsed.session_path else DEFAULT_SESSION_PATH
    loaded = load_session(session_path)
    session = loaded if loaded is not None and loaded.base_url == parsed.base_url else None

    agent_id = parsed.agent_id or (session.agent_id if session is not None else None)

    missing = [
        name
        for name, value in (
            ("--agent-id", agent_id),
            ("--owner-withdraw-private-key", parsed.owner_withdraw_private_key),
        )
        if not value
    ]
    if missing:
        parser.error(f"missing required input(s): {', '.join(missing)}")

    client = CodePitClient(parsed.base_url)
    response = request_reward_withdrawal(
        client,
        agent_id=agent_id,
        amount_raw=parsed.amount_raw,
        client_withdrawal_id=parsed.client_withdrawal_id,
        owner_private_key=parsed.owner_withdraw_private_key,
    )
    print(json.dumps(response, indent=2, default=str))

if __name__ == "__main__":
    main()
