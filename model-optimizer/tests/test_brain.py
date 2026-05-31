"""Tests for ``codepit_optimizer.brain``.

Live-call tests (gated on ``CODEPIT_BRAIN_LIVE_TEST=true``) are NOT
included by default — they would burn provider credit on every CI run.
The tests here use ``Brain.with_stub_responses(...)`` so the LLM seam is
exercised without network.
"""

from __future__ import annotations

import json
import os

import httpx
import pytest

from codepit_optimizer.brain import Brain, BrainConfig, BrainError, RecipeChoice
from codepit_optimizer.brain_providers.groq import GroqBrainProvider
from codepit_optimizer.brain_providers.managed import (
    ManagedBrainError,
    ManagedBrainProvider,
)
from codepit_optimizer.brain_providers.openai import OpenAiBrainProvider
from codepit_optimizer.brain_providers.together import TogetherBrainProvider
from codepit_optimizer.prompts import (
    KNOWN_RECIPE_NAMES,
    OPTIMIZATION_PLAN_SCHEMA,
    RECIPE_CHOICE_SCHEMA,
    build_optimization_plan_prompt,
    build_pick_recipe_prompt,
)


# ---------------------------------------------------------------------------
# BrainConfig
# ---------------------------------------------------------------------------


def test_brain_config_defaults() -> None:
    cfg = BrainConfig()
    assert cfg.tier == "cheap"
    assert cfg.provider_name == "managed"
    assert cfg.max_retries_per_step == 2
    assert cfg.fallback_on_error is True
    assert cfg.action_id_prefix is None


def test_brain_config_rejects_unknown_tier() -> None:
    with pytest.raises(BrainError):
        BrainConfig(tier="ultra")  # type: ignore[arg-type]


def test_brain_config_rejects_negative_retries() -> None:
    with pytest.raises(BrainError):
        BrainConfig(max_retries_per_step=-1)


def test_brain_config_rejects_blank_action_prefix() -> None:
    with pytest.raises(BrainError):
        BrainConfig(action_id_prefix=" ")


# ---------------------------------------------------------------------------
# pick_recipe — happy paths
# ---------------------------------------------------------------------------


def test_pick_recipe_returns_known_recipe_from_structured_response() -> None:
    brain = Brain.with_stub_responses(
        [
            {
                "recipe_name": "graph-optimization",
                "confidence": 0.8,
                "reasoning": "ORT graph opt usually clears the floor",
            },
        ],
    )
    choice = brain.pick_recipe(challenge_spec={"challenge_id": "ch_1"})
    assert isinstance(choice, RecipeChoice)
    assert choice.recipe_name == "graph-optimization"
    assert choice.confidence == 0.8
    assert "ORT" in choice.reasoning


def test_pick_recipe_returns_known_recipe_from_raw_string_response() -> None:
    brain = Brain.with_stub_responses(
        [json.dumps({"recipe_name": "dynamic-int8"})],
    )
    choice = brain.pick_recipe(challenge_spec={"challenge_id": "ch_2"})
    assert choice.recipe_name == "dynamic-int8"
    assert choice.confidence == 0.0


# ---------------------------------------------------------------------------
# pick_recipe — fallback paths
# ---------------------------------------------------------------------------


def test_pick_recipe_falls_back_on_unknown_recipe() -> None:
    brain = Brain.with_stub_responses(
        [{"recipe_name": "magic-quantum-opt", "confidence": 1.0}],
    )
    choice = brain.pick_recipe(challenge_spec={"challenge_id": "ch_3"})
    assert choice.recipe_name == "baseline-export"
    assert "rejected" in choice.reasoning.lower()


def test_pick_recipe_falls_back_on_non_json_output() -> None:
    brain = Brain.with_stub_responses(["not actually json"])
    choice = brain.pick_recipe(challenge_spec={"challenge_id": "ch_4"})
    assert choice.recipe_name == "baseline-export"
    assert "rejected" in choice.reasoning.lower()


def test_pick_recipe_falls_back_on_provider_error() -> None:
    class _Boom:
        def generate(self, **_: object) -> str:
            raise RuntimeError("provider boom")

    brain = Brain(config=BrainConfig(), provider=_Boom())
    choice = brain.pick_recipe(challenge_spec={"challenge_id": "ch_5"})
    assert choice.recipe_name == "baseline-export"
    assert "boom" in choice.reasoning.lower()


def test_pick_recipe_strict_mode_raises_on_provider_error() -> None:
    class _Boom:
        def generate(self, **_: object) -> str:
            raise RuntimeError("provider boom")

    brain = Brain(config=BrainConfig(fallback_on_error=False), provider=_Boom())
    with pytest.raises(BrainError, match="provider boom"):
        brain.pick_recipe(challenge_spec={"challenge_id": "ch_strict"})


def test_pick_recipe_strict_mode_raises_on_bad_output() -> None:
    brain = Brain.with_stub_responses(
        ["not actually json"],
        config=BrainConfig(fallback_on_error=False),
    )
    with pytest.raises(BrainError, match="non-JSON"):
        brain.pick_recipe(challenge_spec={"challenge_id": "ch_bad_output"})


def test_pick_recipe_passes_history_into_prompt() -> None:
    captured: list[str] = []

    class _Capture:
        def generate(self, *, prompt: str, **_: object) -> str:
            captured.append(prompt)
            return json.dumps({"recipe_name": "graph-optimization"})

    brain = Brain(config=BrainConfig(), provider=_Capture())
    brain.pick_recipe(
        challenge_spec={"challenge_id": "ch_6"},
        history=[{"recipe": "baseline-export", "outcome": "failure"}],
    )
    assert len(captured) == 1
    assert "ch_6" in captured[0]
    assert "baseline-export" in captured[0]
    assert "failure" in captured[0]


def test_pick_recipe_prefixes_action_id() -> None:
    captured: list[str] = []

    class _Capture:
        def generate(self, *, action_id: str, **_: object) -> str:
            captured.append(action_id)
            return json.dumps({"recipe_name": "dynamic-int8"})

    brain = Brain(
        config=BrainConfig(action_id_prefix="run-123"),
        provider=_Capture(),
    )
    choice = brain.pick_recipe(challenge_spec={"challenge_id": "ch_prefixed"})
    assert choice.recipe_name == "dynamic-int8"
    assert captured == ["run-123:pick_recipe-0001"]


# ---------------------------------------------------------------------------
# plan_optimization
# ---------------------------------------------------------------------------


def test_plan_optimization_returns_bounded_experiment_plan() -> None:
    brain = Brain.with_stub_responses(
        [
            {
                "objective": "minimize_latency_preserve_quality",
                "strategy": "Try graph optimization followed by quantization.",
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
                "fallback_strategy": "Use graph-only if quantization fails.",
                "stop_rules": {"submit_first_verified_improvement": False},
            },
        ],
    )

    plan = brain.plan_optimization(
        challenge_spec={"challenge_id": "ch_plan", "model_class": "encoder-text-small"},
    )

    assert plan.objective == "minimize_latency_preserve_quality"
    assert plan.experiments[0].name == "o2-int8"
    assert [t.kind for t in plan.experiments[0].executable_transforms] == [
        "onnx_export",
        "dynamic_quantization",
    ]


def test_plan_optimization_accepts_legacy_recipe_response() -> None:
    brain = Brain.with_stub_responses(
        [{"recipe_name": "dynamic-int8", "reasoning": "legacy response"}],
    )

    plan = brain.plan_optimization(challenge_spec={"challenge_id": "ch_legacy"})

    assert plan.legacy_recipe_name == "dynamic-int8"
    assert plan.experiments[0].name == "dynamic-int8"


def test_plan_optimization_falls_back_on_bad_output() -> None:
    brain = Brain.with_stub_responses(["not json"])

    plan = brain.plan_optimization(challenge_spec={"challenge_id": "ch_bad_plan"})

    assert plan.legacy_recipe_name == "baseline-export"
    assert "rejected" in plan.experiments[0].hypothesis


def test_plan_optimization_strict_mode_raises_on_bad_output() -> None:
    brain = Brain.with_stub_responses(
        ["not json"],
        config=BrainConfig(fallback_on_error=False),
    )

    with pytest.raises(BrainError, match="brain plan rejected"):
        brain.plan_optimization(challenge_spec={"challenge_id": "ch_bad_plan"})


def test_plan_optimization_passes_schema_and_prefixed_action_id() -> None:
    captured: list[dict] = []

    class _Capture:
        def generate(self, **kwargs) -> str:
            captured.append(kwargs)
            return json.dumps({"recipe_name": "graph-optimization"})

    brain = Brain(
        config=BrainConfig(action_id_prefix="run-456"),
        provider=_Capture(),
    )
    brain.plan_optimization(
        challenge_spec={"challenge_id": "ch_schema"},
        history=[{"experiment_name": "baseline-export", "outcome": "failure"}],
    )

    assert captured[0]["action_id"] == "run-456:plan_optimization-0001"
    assert captured[0]["schema"] == OPTIMIZATION_PLAN_SCHEMA
    assert "baseline-export" in captured[0]["prompt"]


# ---------------------------------------------------------------------------
# interpret_result
# ---------------------------------------------------------------------------


def test_interpret_result_marks_verified_as_success() -> None:
    brain = Brain.with_stub_responses([])
    summary = brain.interpret_result(
        {
            "state": "VERIFIED",
            "submission_id": "sub_1",
            "proof_record_id": "pr_1",
        },
    )
    assert summary["outcome"] == "success"
    assert summary["state"] == "VERIFIED"
    assert summary["submission_id"] == "sub_1"
    assert summary["proof_record_id"] == "pr_1"


def test_interpret_result_marks_validation_failed_as_failure() -> None:
    brain = Brain.with_stub_responses([])
    summary = brain.interpret_result({"state": "VALIDATION_FAILED"})
    assert summary["outcome"] == "failure"


def test_interpret_result_marks_invalidated_as_cancelled() -> None:
    brain = Brain.with_stub_responses([])
    summary = brain.interpret_result({"state": "INVALIDATED"})
    assert summary["outcome"] == "cancelled"


def test_interpret_result_marks_unknown_state_as_unknown() -> None:
    brain = Brain.with_stub_responses([])
    summary = brain.interpret_result({"state": "WHATEVER"})
    assert summary["outcome"] == "unknown"


# ---------------------------------------------------------------------------
# should_retry
# ---------------------------------------------------------------------------


def test_should_retry_caps_at_max_retries() -> None:
    brain = Brain.with_stub_responses(
        [],
        config=BrainConfig(max_retries_per_step=2),
    )
    assert brain.should_retry(attempt=2, error_code="some_code", retryable=True) is True
    assert brain.should_retry(attempt=3, error_code="some_code", retryable=True) is False


def test_should_retry_honors_explicit_retryable_flag() -> None:
    brain = Brain.with_stub_responses([])
    assert brain.should_retry(attempt=1, error_code=None, retryable=True) is True
    assert brain.should_retry(attempt=1, error_code=None, retryable=False) is False


def test_should_retry_refuses_terminal_codes() -> None:
    brain = Brain.with_stub_responses([])
    assert (
        brain.should_retry(
            attempt=1, error_code="brain.invalid_input", retryable=None,
        )
        is False
    )
    assert (
        brain.should_retry(
            attempt=1, error_code="brain.insufficient_balance", retryable=None,
        )
        is False
    )
    assert (
        brain.should_retry(attempt=1, error_code="auth.invalid_signature", retryable=None)
        is False
    )


def test_should_retry_retries_on_transient_codes() -> None:
    brain = Brain.with_stub_responses([])
    assert (
        brain.should_retry(
            attempt=1, error_code="brain.provider_unavailable", retryable=None,
        )
        is True
    )


# ---------------------------------------------------------------------------
# prompts + schema
# ---------------------------------------------------------------------------


def test_known_recipe_names_match_recipe_choice_schema_enum() -> None:
    schema_enum = RECIPE_CHOICE_SCHEMA["properties"]["recipe_name"]["enum"]
    assert tuple(schema_enum) == KNOWN_RECIPE_NAMES


def test_build_pick_recipe_prompt_contains_recipe_names_and_challenge() -> None:
    prompt = build_pick_recipe_prompt(
        challenge_spec={"challenge_id": "ch_X", "model_class": "encoder-text-small"},
        history=[],
    )
    for name in KNOWN_RECIPE_NAMES:
        assert name in prompt
    assert "ch_X" in prompt
    assert "encoder-text-small" in prompt


def test_build_optimization_plan_prompt_exposes_experiment_controls() -> None:
    prompt = build_optimization_plan_prompt(
        challenge_spec={"challenge_id": "ch_plan_X", "model_class": "encoder-text-small"},
        history=[{"experiment_name": "o2-int8", "outcome": "quality_below_floor"}],
    )

    assert "ch_plan_X" in prompt
    assert "encoder-text-small" in prompt
    assert "o2-int8" in prompt
    assert "onnx_export" in prompt
    assert "dynamic_quantization" in prompt
    assert "package_layout" in prompt
    assert "official verifier" in prompt


# ---------------------------------------------------------------------------
# BYOK provider stubs are wired but inert in V1
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory",
    [GroqBrainProvider, OpenAiBrainProvider, TogetherBrainProvider],
)
def test_byok_providers_raise_not_implemented_in_v1(factory: type) -> None:
    provider = factory(api_key="test")  # type: ignore[call-arg]
    with pytest.raises(NotImplementedError):
        provider.generate(
            prompt="hi",
            action_id="a",
            attempt=1,
            tier="cheap",
        )


# ---------------------------------------------------------------------------
# ManagedBrainProvider — HTTP wiring (mocked transport)
# ---------------------------------------------------------------------------


def test_managed_provider_posts_request_and_returns_content() -> None:
    captured_requests: list[httpx.Request] = []

    def transport_handler(request: httpx.Request) -> httpx.Response:
        captured_requests.append(request)
        return httpx.Response(
            200,
            json={
                "content": "managed-response-text",
                "tokens_in": 10,
                "tokens_out": 5,
                "latency_ms": 7,
                "tier": "cheap",
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
                "cost_codepit": "0",
                "cost_usd_micro": "1",
                "metering_enabled": False,
                "meter_status": "applied",
            },
        )

    transport = httpx.MockTransport(transport_handler)
    client = httpx.Client(transport=transport)
    provider = ManagedBrainProvider(
        base_url="http://engine.test",
        bearer_token="test-secret",
        client=client,
    )
    out = provider.generate(
        prompt="hello",
        action_id="act-1",
        attempt=1,
        tier="cheap",
    )
    assert out == "managed-response-text"
    assert len(captured_requests) == 1
    request = captured_requests[0]
    assert request.method == "POST"
    assert request.url.path == "/v2/brain/generate"
    assert request.headers["authorization"] == "Bearer test-secret"
    body = json.loads(request.content.decode())
    assert body["action_id"] == "act-1"
    assert body["tier"] == "cheap"
    assert body["prompt"] == "hello"


def test_managed_provider_raises_managed_error_on_402() -> None:
    def transport_handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            402,
            json={
                "error": {
                    "code": "brain.insufficient_balance",
                    "message": "out of balance",
                },
                "request_id": "rq_1",
            },
        )

    transport = httpx.MockTransport(transport_handler)
    client = httpx.Client(transport=transport)
    provider = ManagedBrainProvider(
        base_url="http://engine.test",
        bearer_token="test-secret",
        client=client,
    )
    with pytest.raises(ManagedBrainError) as exc:
        provider.generate(prompt="x", action_id="a", attempt=1, tier="cheap")
    assert exc.value.status_code == 402
    assert exc.value.code == "brain.insufficient_balance"


def test_managed_provider_rejects_empty_config() -> None:
    with pytest.raises(ValueError):
        ManagedBrainProvider(base_url="", bearer_token="x")
    with pytest.raises(ValueError):
        ManagedBrainProvider(base_url="http://x", bearer_token="")


# ---------------------------------------------------------------------------
# Live-call gate (skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    os.environ.get("CODEPIT_BRAIN_LIVE_TEST") != "true",
    reason="live brain test gate not enabled",
)
def test_live_managed_provider_call() -> None:  # pragma: no cover - gated
    base_url = os.environ.get("CODEPIT_BRAIN_LIVE_BASE_URL")
    bearer = os.environ.get("CODEPIT_BRAIN_LIVE_BEARER")
    if not base_url or not bearer:
        pytest.skip("missing CODEPIT_BRAIN_LIVE_BASE_URL/_BEARER")
    provider = ManagedBrainProvider(base_url=base_url, bearer_token=bearer)
    out = provider.generate(
        prompt="Reply with the word OK.",
        action_id="live-1",
        attempt=1,
        tier="cheap",
    )
    assert "OK" in out.upper()
