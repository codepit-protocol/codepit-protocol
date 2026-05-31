"""External-agent tiny-chat (ollama-gguf-local) run path against a fake engine.

Reuses the orchestrator's HTTP-boundary fake (``FakeEngine``) so the tiny-chat
entry drives the same canonical register -> discover -> submit -> upload -> poll
loop as the ONNX path, only swapping in a real GGUF build + the tiny-chat
bundle assembler. The GGUF build is an injected seam here (a test double that
emits GGUF-magic bytes); the live llama.cpp build is exercised in task 3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

import pytest

from codepit_optimizer.brain import Brain, BrainConfig
from codepit_optimizer.orchestrator import (
    OrchestratorError,
    TinyChatRunConfig,
    run_tiny_chat_external_agent,
)
from test_orchestrator import FakeEngine, _stub_client

_GGUF_BYTES = b"GGUF" + b"\x00" * 32


def _fake_gguf_builder(
    *,
    calls: list[dict[str, Any]] | None = None,
    gguf_bytes: bytes = _GGUF_BYTES,
) -> Callable[..., dict[str, Any]]:
    """A GGUF-build seam double: writes GGUF-magic bytes and records its call."""

    def builder(*, base_model_ref: str, quantization_profile: str, out_gguf: Path) -> dict[str, Any]:
        if calls is not None:
            calls.append({"base_model_ref": base_model_ref, "quantization_profile": quantization_profile})
        out_gguf.write_bytes(gguf_bytes)
        return {
            "schema": "codepit.tiny_chat.gguf_build.v1",
            "quant_type": quantization_profile.upper(),
            "base_model_ref": base_model_ref,
        }

    return builder


def _tiny_chat_engine() -> FakeEngine:
    engine = FakeEngine()
    engine.challenge_artifact_lane = "ollama-gguf-local"
    return engine


def _config(tmp_path: Path, **overrides: Any) -> TinyChatRunConfig:
    base = dict(
        base_url="http://engine.fake",
        work_dir=tmp_path / "tiny-chat",
        challenge_id="challenge_1",
        session_path=tmp_path / "agent.json",
        poll_interval_s=0.0,
        base_model_ref="hf://Qwen/Qwen2.5-0.5B-Instruct",
        quantization_profile="q4_k_m",
        gguf_build=_fake_gguf_builder(),
    )
    base.update(overrides)
    return TinyChatRunConfig(**base)


def test_tiny_chat_external_run_reaches_verified(tmp_path: Path) -> None:
    engine = _tiny_chat_engine()
    factory = _stub_client(engine)
    calls: list[dict[str, Any]] = []
    config = _config(tmp_path, gguf_build=_fake_gguf_builder(calls=calls))

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_tiny_chat_external_agent(config)

    assert result.state == "VERIFIED"
    assert result.agent_id == "agent_pyopt_1"
    assert result.challenge_id == "challenge_1"

    # the agent built the GGUF on its own compute, for the requested base + profile
    assert calls == [{"base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct", "quantization_profile": "q4_k_m"}]

    # registration (no new auth path) happened before submission
    request_paths = [str(r.url.path) for r in engine.requests]
    assert request_paths[0] == "/v1/agents/auth/challenge"
    assert request_paths[1] == "/v1/agents/register"

    # the full ollama-gguf-local bundle was uploaded, GGUF bytes verbatim
    assert set(engine.uploaded) == {"tiny-chat.gguf", "Modelfile", "provenance.json", "checksums.json"}
    assert engine.uploaded["tiny-chat.gguf"] == _GGUF_BYTES

    # the submitted manifest is the gguf lane
    import json

    submission_body = json.loads(
        next(r for r in engine.requests if r.url.path == "/v1/submissions").read()
    )
    assert submission_body["manifest_envelope"]["artifact_lane"] == "ollama-gguf-local"


def test_tiny_chat_external_run_lets_brain_choose_qwen_quantization(tmp_path: Path) -> None:
    engine = _tiny_chat_engine()
    factory = _stub_client(engine)
    calls: list[dict[str, Any]] = []
    brain = Brain.with_stub_responses(
        [
            {
                "quantization_profile": "q5_k_m",
                "confidence": 0.82,
                "rationale": "Q5_K_M should preserve more benchmark quality than Q4 while staying compact.",
            }
        ],
        config=BrainConfig(
            provider_name="managed",
            tier="premium",
            fallback_on_error=False,
            action_id_prefix="tiny-chat-test",
        ),
    )
    config = _config(tmp_path, brain=brain, gguf_build=_fake_gguf_builder(calls=calls))

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_tiny_chat_external_agent(config)

    assert result.state == "VERIFIED"
    assert result.chosen_recipe == "q5_k_m"
    assert calls == [{"base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct", "quantization_profile": "q5_k_m"}]

    import json

    submission_body = json.loads(
        next(r for r in engine.requests if r.url.path == "/v1/submissions").read()
    )
    assert submission_body["manifest_envelope"]["optimization"]["methods"] == ["q5_k_m"]
    provenance = json.loads(engine.uploaded["provenance.json"].decode("utf-8"))
    assert provenance["quantization_profile"] == "q5_k_m"


def test_resubmitting_same_inputs_is_idempotent(tmp_path: Path) -> None:
    engine = _tiny_chat_engine()
    factory = _stub_client(engine)

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        first = run_tiny_chat_external_agent(_config(tmp_path))
        second = run_tiny_chat_external_agent(_config(tmp_path))

    # a stable client_submission_id is derived, so the engine dedups to one submission
    assert first.client_submission_id == second.client_submission_id
    assert first.submission_id == second.submission_id
    assert len(engine.submissions_by_client_id) == 1


def test_prebuilt_gguf_path_is_submitted_without_invoking_a_builder(tmp_path: Path) -> None:
    engine = _tiny_chat_engine()
    factory = _stub_client(engine)
    calls: list[dict[str, Any]] = []
    prebuilt = tmp_path / "agent-built.gguf"
    prebuilt.write_bytes(_GGUF_BYTES)
    config = _config(tmp_path, gguf_path=prebuilt, gguf_build=_fake_gguf_builder(calls=calls))

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        result = run_tiny_chat_external_agent(config)

    assert result.state == "VERIFIED"
    # the agent supplied its own real GGUF; the build seam is not invoked
    assert calls == []
    assert engine.uploaded["tiny-chat.gguf"] == _GGUF_BYTES


def test_aborts_when_prebuilt_gguf_is_not_a_gguf_binary(tmp_path: Path) -> None:
    engine = _tiny_chat_engine()
    factory = _stub_client(engine)
    bogus = tmp_path / "not-a-model.gguf"
    bogus.write_bytes(b"this is not a gguf")
    config = _config(tmp_path, gguf_path=bogus, gguf_build=None)

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with pytest.raises(OrchestratorError, match="GGUF"):
            run_tiny_chat_external_agent(config)

    assert "/v1/submissions" not in [str(r.url.path) for r in engine.requests]


def test_aborts_when_challenge_is_not_the_gguf_lane(tmp_path: Path) -> None:
    engine = FakeEngine()  # default lane is onnx-browser-webgpu
    factory = _stub_client(engine)
    calls: list[dict[str, Any]] = []
    config = _config(tmp_path, gguf_build=_fake_gguf_builder(calls=calls))

    with patch("codepit_optimizer.orchestrator.CodePitClient", side_effect=factory):
        with pytest.raises(OrchestratorError, match="ollama-gguf-local"):
            run_tiny_chat_external_agent(config)

    # the guard fires before any GGUF is built or submitted
    assert calls == []
    assert "/v1/submissions" not in [str(r.url.path) for r in engine.requests]
