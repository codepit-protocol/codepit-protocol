"""End-to-end orchestrator tests with a fake engine.

We stub the HTTP layer with ``httpx.MockTransport`` and the recipe layer
with a fake runner that drops a known bundle into the work dir. This
exercises the full orchestrator decision tree (register, eligibility,
candidate selection, manifest, upload, polling) without touching a real
engine — the live-engine validation is a separate task.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Mapping
from unittest.mock import patch

import httpx
import pytest

from codepit_optimizer.brain import Brain, BrainConfig, BrainError
from codepit_optimizer.orchestrator import (
    ForeverConfig,
    OLLAMA_GGUF_LOCAL_ARTIFACT_LANE,
    OrchestratorConfig,
    OrchestratorError,
    TinyChatRunConfig,
    _assert_payout_bound_for_reward,
    _resolve_tiny_chat_challenge,
    build_client_submission_id,
    run_optimizer_agent,
    run_optimizer_agent_forever,
)
from codepit_optimizer.payload_hash import hash_registration_payload
from codepit_optimizer.protocol import CodePitClient


# ---------------------------------------------------------------------------
# Fake engine
# ---------------------------------------------------------------------------


class FakeEngine:
    """Tiny in-memory implementation of the V2 protocol for orchestrator tests.

    Records every request so tests can assert ordering and payload shapes.
    Returns presigned URLs at ``http://uploads.fake/<logical_name>`` that
    the same handler accepts.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.uploaded: dict[str, bytes] = {}
        self.terminal_state = "VERIFIED"
        self.next_eligible = True
        self.eligibility_reasons: list[str] = []
        self.poll_responses: list[str] = ["QUEUED_FOR_BENCHMARK", "BENCHMARKING"]
        self.benchmark_target_version = "0.1.0"
        # The challenge snapshot's lane (GET /v1/challenges/:id). Defaults to the
        # ONNX lane these tests exercise; the tiny-chat run path sets it to
        # ollama-gguf-local. run_optimizer_agent ignores this field.
        self.challenge_artifact_lane = "onnx-browser-webgpu"
        self.submissions_by_client_id: dict[str, str] = {}
        self.public_baseline_comparison: dict | None = {
            "improved": True,
            "quality_floor_met": True,
        }
        self.balance_read_timeouts_remaining = 0
        self.registration_payload_hash: str | None = None
        self.registration_sybil_gate: dict[str, Any] | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = request.url
        path = url.path
        method = request.method

        if str(url).startswith("http://uploads.fake/"):
            self.uploaded[url.path.lstrip("/")] = request.read()
            return httpx.Response(200)

        if method == "POST" and path == "/v1/agents/auth/challenge":
            body = json.loads(request.read() or b"{}")
            self.registration_payload_hash = body["registration_payload_hash"]
            response = {
                "challenge_id": "ch_auth_1",
                "nonce": "nonce_1",
                "message": "please sign",
                "expires_at": "2026-05-01T00:01:00Z",
            }
            if self.registration_sybil_gate is not None:
                response["sybil_gate"] = self.registration_sybil_gate
            return httpx.Response(
                200,
                json=response,
            )

        if method == "POST" and path == "/v1/agents/register":
            body = json.loads(request.read() or b"{}")
            expected_hash = hash_registration_payload(
                {
                    "protocol_version": body["protocol_version"],
                    "agent_signer_address": body["agent_signer_address"],
                    "agent": body["agent"],
                    "capabilities": body["capabilities"],
                    "agent_wallet": body["agent_wallet"],
                }
            )
            assert self.registration_payload_hash == expected_hash
            if self.registration_sybil_gate is not None:
                solution = body.get("sybil_gate_solution")
                assert solution is not None
                assert solution["kind"] == "hashcash"
                digest = hashlib.sha256(
                    ":".join(
                        [
                            "codepit:v2:registration-pow",
                            body["agent_signer_address"].lower(),
                            expected_hash,
                            "nonce_1",
                            solution["nonce"],
                        ]
                    ).encode("utf-8")
                ).hexdigest()
                assert _count_leading_zero_bits(digest) >= self.registration_sybil_gate[
                    "difficulty_bits"
                ]
            return httpx.Response(
                201,
                json={
                    "agent_id": "agent_pyopt_1",
                    "trust_tier": "Sandbox",
                    "credential": {"id": "cred_1", "secret": "rt_secret_xyz"},
                },
            )

        if method == "GET" and path == "/v1/challenges/next":
            return httpx.Response(
                200,
                json={"challenge": {"challenge_id": "challenge_1"}},
            )

        if method == "GET" and path == "/v1/challenges/challenge_1":
            return httpx.Response(
                200,
                json={
                    "challenge_id": "challenge_1",
                    "benchmark_target_version": self.benchmark_target_version,
                    "artifact_lane": self.challenge_artifact_lane,
                },
            )

        eligibility_path = "/v1/agents/agent_pyopt_1/eligibility"
        if method == "GET" and path == eligibility_path:
            return httpx.Response(
                200,
                json={"eligible": self.next_eligible, "reasons": self.eligibility_reasons},
            )

        if method == "POST" and path == "/v1/submissions":
            body = json.loads(request.read())
            files = body["manifest_envelope"]["files"]
            client_submission_id = body["client_submission_id"]
            submission_id = self.submissions_by_client_id.get(client_submission_id)
            if submission_id is None:
                submission_id = f"sub_{len(self.submissions_by_client_id) + 1}"
                self.submissions_by_client_id[client_submission_id] = submission_id
            return httpx.Response(
                201,
                json={
                    "submission_id": submission_id,
                    "state": "CREATED",
                    "upload_orchestration": {
                        "kind": "presigned-urls",
                        "expires_at": (
                            datetime.now(timezone.utc) + timedelta(minutes=15)
                        ).isoformat().replace("+00:00", "Z"),
                        "files": [
                            {
                                "logical_name": file["logical_name"],
                                "upload_url": f"http://uploads.fake/{file['logical_name']}",
                                "media_type": file["media_type"],
                                "size_bytes": file["size_bytes"],
                                "sha256": file["sha256"],
                            }
                            for file in files
                        ],
                    },
                },
            )

        if method == "GET" and path.startswith("/v1/submissions/"):
            submission_id = path.rsplit("/", 1)[-1]
            if self.poll_responses:
                state = self.poll_responses.pop(0)
            else:
                state = self.terminal_state
            return httpx.Response(
                200,
                json={
                    "submission_id": submission_id,
                    "state": state,
                    "proof_record_id": "proof_1" if state == "VERIFIED" else None,
                    "upload_summary": {"files": []},
                },
            )

        if method == "GET" and path.startswith("/api/v2/public/submissions/"):
            submission_id = path.rsplit("/", 1)[-1]
            suffix = submission_id.split("_")[-1]
            return httpx.Response(
                200,
                json={
                    "submission_id": submission_id,
                    "lifecycle_state": self.terminal_state,
                    "benchmark_state": {
                        "result_id": f"res_{suffix}",
                        "proof_record_id": "proof_1",
                        "proof_record_status": "PREPARED",
                    },
                },
            )

        if method == "GET" and path.startswith("/api/v2/public/results/"):
            result_id = path.rsplit("/", 1)[-1]
            suffix = result_id.split("_")[-1]
            return httpx.Response(
                200,
                json={
                    "result_id": result_id,
                    "submission_id": f"sub_{suffix}",
                    "proof_record_id": "proof_1",
                    "proof_record_status": "PREPARED",
                    "metrics": {
                        "pass": True,
                        "latency_us": 100,
                        "memory_bytes": 2048,
                        "artifact_size_bytes": 512,
                        "quality_score": 0.95,
                    },
                    "baseline_comparison": self.public_baseline_comparison,
                },
            )

        if method == "GET" and path == "/v1/agents/agent_pyopt_1/balances":
            if self.balance_read_timeouts_remaining > 0:
                self.balance_read_timeouts_remaining -= 1
                raise httpx.ReadTimeout("engine verifier is busy")
            return httpx.Response(
                200,
                json={
                    "internal_balance": "1000",
                    "spendable_balance": "1000",
                    "locked_balance": "0",
                    "working_balance": "1000",
                },
            )

        if method == "GET" and path == "/v1/agents/agent_pyopt_1/rewards":
            return httpx.Response(
                200,
                json={"pending_total": "0", "settled_total": "0", "recent_events": []},
            )

        return httpx.Response(404, json={"error": {"code": "not_found", "message": path}})


def _stub_client(engine: FakeEngine) -> Callable[[str], CodePitClient]:
    """Return a factory that builds a CodePitClient backed by the fake engine."""

    transport = httpx.MockTransport(engine.handler)

    def factory(base_url: str, agent_id: str | None = None, credential: str | None = None) -> CodePitClient:
        return CodePitClient(
            base_url,
            agent_id=agent_id,
            credential=credential,
            transport=transport,
        )

    return factory


def _seed_bundle(work_dir: Path, recipe_name: str = "graph-optimization") -> Path:
    target = work_dir / recipe_name
    target.mkdir(parents=True, exist_ok=True)
    (target / "model.onnx").write_bytes(b"\x01onnx-bytes")
    (target / "config.json").write_bytes(b'{"hidden":1}')
    return target


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_full_flow_against_fake_engine_reaches_verified(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        pre_built_bundle_dir=bundle_dir,
        session_path=tmp_path / "agent.json",
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_optimizer_agent(config)

    assert result.state == "VERIFIED"
    assert result.agent_id == "agent_pyopt_1"
    assert result.proof_record_id == "proof_1"
    assert result.result_id == "res_1"
    assert result.receipt_path == "/receipts/res_1"
    assert result.verified_improvement is True
    assert result.baseline_comparison == {"improved": True, "quality_floor_met": True}
    assert result.balances["internal_balance"] == "1000"
    assert result.reused_session is False
    assert (tmp_path / "agent.json").exists()
    from codepit_optimizer.session import load_session

    saved = load_session(tmp_path / "agent.json")
    assert saved is not None
    assert saved.agent_wallet_private_key
    assert saved.agent_wallet_address

    request_paths = [str(request.url.path) for request in engine.requests]
    # registration first, then discovery, then submission
    assert request_paths[0] == "/v1/agents/auth/challenge"
    assert request_paths[1] == "/v1/agents/register"
    register_body = json.loads(engine.requests[1].read())
    assert register_body["agent_wallet"]["wallet_provider"] == "local"
    assert register_body["agent_wallet"]["custody_mode"] == "agent_local"
    assert register_body["agent_wallet"]["address"] == saved.agent_wallet_address
    assert register_body["agent_wallet_signature"].startswith("0x")
    assert isinstance(register_body["agent_wallet_timestamp_ms"], int)
    assert "/v1/submissions" in request_paths
    assert "/v1/submissions/sub_1" in request_paths
    # uploads happened
    assert set(engine.uploaded.keys()) == {"model.onnx", "config.json"}
    assert engine.uploaded["model.onnx"] == b"\x01onnx-bytes"


def test_full_flow_keeps_verified_result_when_final_balance_read_times_out(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.balance_read_timeouts_remaining = 1
    factory = _stub_client(engine)

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        pre_built_bundle_dir=bundle_dir,
        session_path=tmp_path / "agent.json",
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_optimizer_agent(config)

    assert result.state == "VERIFIED"
    assert result.result_id == "res_1"
    assert result.balances == {}
    assert result.rewards["pending_total"] == "0"


def test_registration_solves_hashcash_sybil_gate(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.registration_sybil_gate = {
        "kind": "hashcash",
        "algorithm": "sha256-leading-zero-bits",
        "difficulty_bits": 8,
        "challenge": "auth-challenge-digest",
        "input_format": (
            "sha256('codepit:v2:registration-pow:<lowercase_agent_signer_address>:"
            "<registration_payload_hash>:<auth_challenge_nonce>:<solution_nonce>')"
        ),
    }
    factory = _stub_client(engine)

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        pre_built_bundle_dir=bundle_dir,
        session_path=None,
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_optimizer_agent(config)

    assert result.agent_id == "agent_pyopt_1"
    register_body = json.loads(engine.requests[1].read())
    assert register_body["sybil_gate_solution"]["kind"] == "hashcash"
    assert register_body["sybil_gate_solution"]["nonce"]


def test_receipt_without_comparison_does_not_claim_improvement(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.public_baseline_comparison = None
    factory = _stub_client(engine)

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        pre_built_bundle_dir=bundle_dir,
        session_path=None,
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_optimizer_agent(config)

    assert result.result_id == "res_1"
    assert result.receipt_path == "/receipts/res_1"
    assert result.baseline_comparison is None
    assert result.verified_improvement is False


def test_non_improving_receipt_does_not_claim_improvement(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.public_baseline_comparison = {
        "improved": False,
        "quality_floor_met": True,
    }
    factory = _stub_client(engine)

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        pre_built_bundle_dir=bundle_dir,
        session_path=None,
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_optimizer_agent(config)

    assert result.baseline_comparison == {"improved": False, "quality_floor_met": True}
    assert result.verified_improvement is False


def test_selected_recipe_runs_only_that_recipe_and_marks_manifest(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)
    called_recipe_names: list[str] = []

    def fake_run_candidate_recipes(*, source_model: str, work_dir: Path, recipes):
        results = []
        from codepit_optimizer.recipes import RecipeRunResult

        for recipe in recipes:
            called_recipe_names.append(recipe.name)
            bundle_dir = _seed_bundle(work_dir, recipe.name)
            results.append(
                RecipeRunResult(
                    name=recipe.name,
                    output_dir=bundle_dir,
                    succeeded=True,
                )
            )
        return results

    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        selected_recipe="dynamic-int8",
        session_path=None,
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with patch(
            "codepit_optimizer.orchestrator.run_candidate_recipes",
            side_effect=fake_run_candidate_recipes,
        ):
            result = run_optimizer_agent(config)

    assert result.chosen_recipe == "dynamic-int8"
    assert called_recipe_names == ["dynamic-int8"]
    submission = next(
        request for request in engine.requests if request.url.path == "/v1/submissions"
    )
    body = json.loads(submission.read())
    assert body["manifest_envelope"]["optimization"]["methods"] == ["dynamic-int8"]


def test_strict_brain_mode_fails_when_provider_is_unavailable(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)

    class UnavailableProvider:
        def generate(self, **kwargs) -> str:
            raise RuntimeError("managed brain unavailable")

    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        brain=Brain(
            config=BrainConfig(fallback_on_error=False),
            provider=UnavailableProvider(),
        ),
        session_path=None,
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with pytest.raises(BrainError, match="managed brain unavailable"):
            run_optimizer_agent(config)

    request_paths = [str(request.url.path) for request in engine.requests]
    assert "/v1/submissions" not in request_paths


def test_successful_run_records_brain_optimization_plan(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)

    def fake_run_plan_experiments(*, plan, source_model: str, work_dir: Path):
        from codepit_optimizer.recipes import RecipeRunResult

        return [
            RecipeRunResult(
                name=experiment.name,
                output_dir=_seed_bundle(work_dir, experiment.name),
                succeeded=True,
            )
            for experiment in plan.experiments
        ]

    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        brain=Brain.with_stub_responses(
            [
                {
                    "objective": "minimize_latency_preserve_quality",
                    "strategy": "Try graph optimized int8 first.",
                    "max_experiments": 1,
                    "experiments": [
                        {
                            "name": "o2-int8",
                            "hypothesis": "O2 plus int8 should improve latency.",
                            "transforms": [
                                {"kind": "onnx_export", "optimize": "O2"},
                                {
                                    "kind": "dynamic_quantization",
                                    "weight_type": "qint8",
                                    "per_channel": False,
                                },
                            ],
                            "expected_tradeoff": "possible quality drop",
                        },
                    ],
                },
            ],
        ),
        session_path=None,
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with patch(
            "codepit_optimizer.orchestrator.run_plan_experiments",
            side_effect=fake_run_plan_experiments,
        ):
            result = run_optimizer_agent(config)

    assert result.brain_plan is not None
    assert result.brain_plan.experiments[0].name == "o2-int8"
    assert result.brain_decision is None
    assert result.chosen_recipe == "o2-int8"


def test_legacy_brain_recipe_response_still_runs_as_plan(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)
    called_plan_names: list[str] = []

    def fake_run_plan_experiments(*, plan, source_model: str, work_dir: Path):
        from codepit_optimizer.recipes import RecipeRunResult

        called_plan_names.extend(experiment.name for experiment in plan.experiments)
        return [
            RecipeRunResult(
                name="graph-optimization",
                output_dir=_seed_bundle(work_dir, "graph-optimization"),
                succeeded=True,
            )
        ]

    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        brain=Brain.with_stub_responses(
            [
                {
                    "recipe_name": "graph-optimization",
                    "confidence": 0.82,
                    "reasoning": "legacy recipe response",
                },
            ],
        ),
        session_path=None,
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with patch(
            "codepit_optimizer.orchestrator.run_plan_experiments",
            side_effect=fake_run_plan_experiments,
        ):
            result = run_optimizer_agent(config)

    assert called_plan_names == ["graph-optimization"]
    assert result.brain_plan is not None
    assert result.brain_plan.legacy_recipe_name == "graph-optimization"
    assert result.brain_decision is not None
    assert result.brain_decision.recipe_name == "graph-optimization"


def test_generated_client_submission_id_matches_manifest_intent(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        pre_built_bundle_dir=bundle_dir,
        session_path=None,
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_optimizer_agent(config)

    submission = next(
        request for request in engine.requests if request.url.path == "/v1/submissions"
    )
    body = json.loads(submission.read())
    expected = build_client_submission_id(
        agent_id="agent_pyopt_1",
        challenge_id="challenge_1",
        manifest_envelope=body["manifest_envelope"],
    )
    assert body["client_submission_id"] == expected
    assert result.client_submission_id == expected
    assert len(expected.encode("utf-8")) <= 128


def test_explicit_client_submission_id_is_sent_unchanged(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        pre_built_bundle_dir=bundle_dir,
        client_submission_id="retry-key-001",
        session_path=None,
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_optimizer_agent(config)

    submission = next(
        request for request in engine.requests if request.url.path == "/v1/submissions"
    )
    body = json.loads(submission.read())
    assert body["client_submission_id"] == "retry-key-001"
    assert result.client_submission_id == "retry-key-001"


def test_invalid_client_submission_id_fails_before_protocol_calls(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        client_submission_id="",
        session_path=None,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with pytest.raises(OrchestratorError, match="client_submission_id"):
            run_optimizer_agent(config)

    assert engine.requests == []


def test_invalid_selected_recipe_fails_before_protocol_calls(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        selected_recipe="not-a-recipe",
        session_path=None,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with pytest.raises(OrchestratorError, match="Valid recipes"):
            run_optimizer_agent(config)

    assert engine.requests == []


def test_reuses_session_when_persisted(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)
    session_path = tmp_path / "agent.json"
    session_path.parent.mkdir(parents=True, exist_ok=True)

    # Pre-seed a session matching what the engine would have returned.
    from codepit_optimizer.session import AgentSession, save_session
    from codepit_optimizer.signer import AgentSigner

    signer = AgentSigner.from_private_key("0x" + "33" * 32)
    save_session(
        AgentSession(
            base_url="http://engine.fake",
            agent_id="agent_pyopt_1",
            signer_private_key=signer.private_key,
            signer_address=signer.address,
            runtime_credential="rt_secret_xyz",
            runtime_credential_id="cred_1",
            trust_tier="Sandbox",
        ),
        path=session_path,
    )

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        pre_built_bundle_dir=bundle_dir,
        session_path=session_path,
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_optimizer_agent(config)

    assert result.reused_session is True
    request_paths = [str(request.url.path) for request in engine.requests]
    # registration endpoints must NOT have been hit
    assert "/v1/agents/auth/challenge" not in request_paths
    assert "/v1/agents/register" not in request_paths
    # but discovery and submission still happened
    assert "/v1/challenges/next" in request_paths
    assert "/v1/submissions" in request_paths


def test_uses_provisioned_managed_session_without_registering(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        private_key="0x" + "44" * 32,
        agent_id="agent_pyopt_1",
        runtime_credential="managed_runtime_secret",
        pre_built_bundle_dir=bundle_dir,
        session_path=tmp_path / "managed-session.json",
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_optimizer_agent(config)

    assert result.state == "VERIFIED"
    assert result.agent_id == "agent_pyopt_1"
    assert result.reused_session is True
    assert (tmp_path / "managed-session.json").exists()
    request_paths = [str(request.url.path) for request in engine.requests]
    assert "/v1/agents/auth/challenge" not in request_paths
    assert "/v1/agents/register" not in request_paths
    assert "/v1/challenges/next" in request_paths
    assert "/v1/submissions" in request_paths


def test_forever_reuses_provisioned_session_without_registering(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)

    bundle_dir = _seed_bundle(tmp_path / "bundle")
    base_config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        private_key="0x" + "44" * 32,
        agent_id="agent_pyopt_1",
        runtime_credential="managed_runtime_secret",
        runtime_credential_id="cred_1",
        pre_built_bundle_dir=bundle_dir,
        session_path=None,
        poll_interval_s=0.0,
    )

    forever = ForeverConfig(
        base_config=base_config,
        max_iterations=1,
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        summary = run_optimizer_agent_forever(forever, install_signal_handlers=False)

    assert summary.stopped_reason == "max_iterations"
    assert summary.iterations_started == 1
    assert summary.iterations_completed == 1
    request_paths = [str(request.url.path) for request in engine.requests]
    assert "/v1/agents/auth/challenge" not in request_paths
    assert "/v1/agents/register" not in request_paths

    submission_request = next(
        request
        for request in engine.requests
        if request.method == "POST" and request.url.path == "/v1/submissions"
    )
    submission_body = json.loads(submission_request.content)
    assert submission_body["agent_id"] == "agent_pyopt_1"


def test_forever_feeds_verified_result_history_back_to_brain(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)

    class RecordingProvider:
        def __init__(self) -> None:
            self.prompts: list[str] = []
            self.responses = [
                {
                    "objective": "learn from first attempt",
                    "strategy": "Try graph cleanup first.",
                    "max_experiments": 1,
                    "experiments": [
                        {
                            "name": "graph-first",
                            "hypothesis": "Graph cleanup should improve the browser bundle.",
                            "transforms": [{"kind": "onnx_export", "optimize": "O2"}],
                        },
                    ],
                },
                {
                    "objective": "react to verifier history",
                    "strategy": "Verifier history was good, now test int8.",
                    "max_experiments": 1,
                    "experiments": [
                        {
                            "name": "int8-after-history",
                            "hypothesis": "Quantization can reduce package size after a passing graph run.",
                            "transforms": [
                                {"kind": "onnx_export", "optimize": "O2"},
                                {
                                    "kind": "dynamic_quantization",
                                    "weight_type": "qint8",
                                    "per_channel": False,
                                },
                            ],
                        },
                    ],
                },
            ]

        def generate(
            self,
            *,
            prompt: str,
            action_id: str,
            attempt: int,
            tier: str,
            schema: Mapping[str, Any] | None = None,
            system: str | None = None,
        ) -> str:
            self.prompts.append(prompt)
            return json.dumps(self.responses.pop(0))

    provider = RecordingProvider()

    def fake_run_plan_experiments(*, plan, source_model: str, work_dir: Path):
        from codepit_optimizer.recipes import RecipeRunResult

        return [
            RecipeRunResult(
                name=experiment.name,
                output_dir=_seed_bundle(work_dir, experiment.name),
                succeeded=True,
            )
            for experiment in plan.experiments
        ]

    base_config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        brain=Brain(config=BrainConfig(), provider=provider),
        session_path=None,
        poll_interval_s=0.0,
    )
    forever = ForeverConfig(
        base_config=base_config,
        max_iterations=2,
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with patch(
            "codepit_optimizer.orchestrator.run_plan_experiments",
            side_effect=fake_run_plan_experiments,
        ):
            summary = run_optimizer_agent_forever(
                forever,
                install_signal_handlers=False,
            )

    assert summary.iterations_completed == 2
    assert len(provider.prompts) == 2
    assert '"chosen_recipe": "graph-first"' in provider.prompts[1]
    assert '"verified_improvement": true' in provider.prompts[1]
    assert '"result_id": "res_1"' in provider.prompts[1]


def test_provisioned_session_requires_agent_id_and_credential_pair(tmp_path: Path) -> None:
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        agent_id="agent_pyopt_1",
        runtime_credential=None,
        session_path=None,
    )

    with pytest.raises(OrchestratorError, match="provided together"):
        run_optimizer_agent(config)


def test_aborts_when_not_eligible(tmp_path: Path) -> None:
    engine = FakeEngine()
    engine.next_eligible = False
    engine.eligibility_reasons = ["agent.suspended"]
    factory = _stub_client(engine)

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        pre_built_bundle_dir=bundle_dir,
        session_path=None,
        poll_interval_s=0.0,
    )

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with pytest.raises(OrchestratorError, match=re.compile("not eligible.*agent.suspended")):
            run_optimizer_agent(config)


def test_aborts_when_recipe_pipeline_returns_no_successes(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        session_path=None,
        poll_interval_s=0.0,
    )

    def fake_run_candidate_recipes(*, source_model: str, work_dir: Path):
        # mimic every recipe failing
        from codepit_optimizer.recipes import RECIPES, RecipeRunResult

        return [
            RecipeRunResult(
                name=recipe.name,
                output_dir=work_dir / recipe.name,
                succeeded=False,
                error="missing dep",
            )
            for recipe in RECIPES
        ]

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with patch(
            "codepit_optimizer.orchestrator.run_candidate_recipes",
            side_effect=fake_run_candidate_recipes,
        ):
            with pytest.raises(OrchestratorError, match="no recipe succeeded"):
                run_optimizer_agent(config)


def test_aborts_on_upload_hash_mismatch(tmp_path: Path) -> None:
    engine = FakeEngine()
    factory = _stub_client(engine)

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        pre_built_bundle_dir=bundle_dir,
        session_path=None,
        poll_interval_s=0.0,
    )

    original_handler = engine.handler

    def lying_handler(request: httpx.Request) -> httpx.Response:
        response = original_handler(request)
        if request.method == "POST" and request.url.path == "/v1/submissions":
            payload = response.json()
            for instruction in payload["upload_orchestration"]["files"]:
                instruction["sha256"] = hashlib.sha256(b"different").hexdigest()
            return httpx.Response(201, json=payload)
        return response

    transport = httpx.MockTransport(lying_handler)

    def liar_factory(base_url, agent_id=None, credential=None):
        return CodePitClient(base_url, agent_id=agent_id, credential=credential, transport=transport)

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=liar_factory):
        with pytest.raises(OrchestratorError, match="hash mismatch"):
            run_optimizer_agent(config)


def test_rejects_presigned_upload_ttl_above_one_hour(tmp_path: Path) -> None:
    engine = FakeEngine()

    bundle_dir = _seed_bundle(tmp_path / "candidates")
    config = OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "candidates",
        pre_built_bundle_dir=bundle_dir,
        session_path=None,
        poll_interval_s=0.0,
    )

    original_handler = engine.handler

    def long_ttl_handler(request: httpx.Request) -> httpx.Response:
        response = original_handler(request)
        if request.method == "POST" and request.url.path == "/v1/submissions":
            payload = response.json()
            payload["upload_orchestration"]["expires_at"] = (
                datetime.now(timezone.utc) + timedelta(hours=2)
            ).isoformat().replace("+00:00", "Z")
            return httpx.Response(201, json=payload)
        return response

    transport = httpx.MockTransport(long_ttl_handler)

    def factory(base_url, agent_id=None, credential=None):
        return CodePitClient(base_url, agent_id=agent_id, credential=credential, transport=transport)

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with pytest.raises(OrchestratorError, match="TTL exceeds"):
            run_optimizer_agent(config)

    assert engine.uploaded == {}


# --------------------------------------------------------------------------
# tiny-chat challenge targeting (slice G, #276): explicit / sponsor / next
# --------------------------------------------------------------------------


class _TargetingClient:
    """Stub exposing only what _resolve_tiny_chat_challenge touches."""

    def __init__(
        self,
        *,
        public_items: list[dict[str, Any]] | None = None,
        eligible_ids: set[str] | None = None,
        next_challenge_id: str | None = None,
    ) -> None:
        self._public_items = public_items or []
        self._eligible_ids = eligible_ids or set()
        self._next_challenge_id = next_challenge_id
        self.calls: list[str] = []

    def list_public_challenges(self) -> dict[str, Any]:
        self.calls.append("list_public_challenges")
        return {"items": self._public_items}

    def read_eligibility(self, challenge_id: str) -> dict[str, Any]:
        self.calls.append(f"read_eligibility:{challenge_id}")
        return {"eligible": challenge_id in self._eligible_ids, "reasons": []}

    def next_challenge(self) -> dict[str, Any]:
        self.calls.append("next_challenge")
        if self._next_challenge_id is None:
            return {"challenge": None}
        return {"challenge": {"challenge_id": self._next_challenge_id}}


def _sponsor_item(challenge_id: str, pool_raw: str) -> dict[str, Any]:
    return {
        "challenge_id": challenge_id,
        "lifecycle_state": "Open",
        "artifact_lane": OLLAMA_GGUF_LOCAL_ARTIFACT_LANE,
        "sponsor_competition": True,
        "bounty_terms": {"total_pool_raw": pool_raw},
    }


def test_resolve_tiny_chat_challenge_explicit_id_wins() -> None:
    client = _TargetingClient(next_challenge_id="should-not-be-used")
    config = TinyChatRunConfig(
        base_url="http://e", work_dir=Path("/tmp"), challenge_id="explicit-ch", target="sponsor",
    )

    assert _resolve_tiny_chat_challenge(client, config) == "explicit-ch"
    # explicit id short-circuits: no discovery, no next-challenge probe
    assert client.calls == []


def test_resolve_tiny_chat_challenge_sponsor_target_discovers_richest_eligible() -> None:
    client = _TargetingClient(
        public_items=[_sponsor_item("rich", "900"), _sponsor_item("mid", "500")],
        eligible_ids={"mid"},
    )
    config = TinyChatRunConfig(base_url="http://e", work_dir=Path("/tmp"), target="sponsor")

    assert _resolve_tiny_chat_challenge(client, config) == "mid"


def test_resolve_tiny_chat_challenge_sponsor_target_raises_when_none_eligible() -> None:
    client = _TargetingClient(public_items=[_sponsor_item("rich", "900")], eligible_ids=set())
    config = TinyChatRunConfig(base_url="http://e", work_dir=Path("/tmp"), target="sponsor")

    with pytest.raises(OrchestratorError, match="sponsor"):
        _resolve_tiny_chat_challenge(client, config)


def test_resolve_tiny_chat_challenge_default_target_uses_next_challenge() -> None:
    client = _TargetingClient(next_challenge_id="next-ch")
    config = TinyChatRunConfig(base_url="http://e", work_dir=Path("/tmp"))  # target defaults

    assert _resolve_tiny_chat_challenge(client, config) == "next-ch"
    assert "next_challenge" in client.calls
    assert "list_public_challenges" not in client.calls


# --------------------------------------------------------------------------
# payout-binding footgun guard (slice F, #272): no silent reward forfeit
# --------------------------------------------------------------------------


class _PayoutReadClient:
    def __init__(self, payout_address: str | None) -> None:
        self._payout_address = payout_address
        self.read_agent_calls = 0

    def read_agent(self) -> dict[str, Any]:
        self.read_agent_calls += 1
        return {"agent_id": "a", "payout_address": self._payout_address}


def test_reward_target_without_bound_payout_is_refused() -> None:
    client = _PayoutReadClient(payout_address=None)
    config = TinyChatRunConfig(base_url="http://e", work_dir=Path("/tmp"), target="sponsor")

    with pytest.raises(OrchestratorError, match="payout"):
        _assert_payout_bound_for_reward(client, config)


def test_reward_target_with_bound_payout_is_allowed() -> None:
    client = _PayoutReadClient(payout_address="0x" + "d" * 40)
    config = TinyChatRunConfig(base_url="http://e", work_dir=Path("/tmp"), target="sponsor")

    _assert_payout_bound_for_reward(client, config)  # does not raise


def test_unbound_payout_override_allows_reward_target() -> None:
    client = _PayoutReadClient(payout_address=None)
    config = TinyChatRunConfig(
        base_url="http://e", work_dir=Path("/tmp"), target="sponsor", allow_unbound_payout=True,
    )

    _assert_payout_bound_for_reward(client, config)  # override: does not raise
    # override short-circuits before any network read
    assert client.read_agent_calls == 0


def test_non_reward_target_skips_payout_guard() -> None:
    client = _PayoutReadClient(payout_address=None)
    config = TinyChatRunConfig(base_url="http://e", work_dir=Path("/tmp"))  # target defaults

    _assert_payout_bound_for_reward(client, config)  # non-sponsor: no guard
    assert client.read_agent_calls == 0
