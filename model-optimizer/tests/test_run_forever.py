"""Run-forever supervisor tests.

We don't run the real ``run_optimizer_agent`` here — that's exercised by
``test_orchestrator.py`` against a fake engine. These tests pin the
supervisor's *control flow* invariants: idle handling, transient retry,
fatal exit, and stop_event responsiveness.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from codepit_optimizer.orchestrator import (
    ForeverConfig,
    ForeverIterationOutcome,
    OrchestratorConfig,
    OrchestratorError,
    OrchestratorResult,
    default_lane_runners,
    run_optimizer_agent_forever,
)
from codepit_optimizer.protocol import ProtocolError


def _base_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        base_url="http://engine.fake",
        work_dir=tmp_path / "work",
        session_path=None,
        poll_interval_s=0.0,
        poll_timeout_s=1.0,
    )


def _result(state: str = "VERIFIED") -> OrchestratorResult:
    return OrchestratorResult(
        agent_id="agt_x",
        signer_address="0x" + "a" * 40,
        challenge_id="ch_x",
        submission_id="sub_x",
        state=state,
        benchmark_target_version="0.1.0",
        chosen_recipe="pre-built",
        bundle_dir=Path("/tmp/dummy"),
    )


def test_lane_runners_dispatch_on_eligible_challenge_lane(tmp_path, monkeypatch):
    """With a lane_runners registry, the supervisor peeks at the next eligible
    challenge's lane and invokes the matching runner — not the legacy encoder
    path. This is the #87 "generic dispatch" guarantee: any agent can claim a
    challenge on any lane it has registered a runner for, without being locked
    to a single recipe at boot time.
    """
    encoder_calls: list[str | None] = []
    tiny_chat_calls: list[str | None] = []

    def fake_encoder(config):
        encoder_calls.append(config.challenge_id)
        return _result(state="VERIFIED")

    def fake_tiny_chat(config):
        tiny_chat_calls.append(config.challenge_id)
        return _result(state="VERIFIED")

    forever = ForeverConfig(
        base_config=_base_config(tmp_path),
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
        max_iterations=1,
        lane_runners={
            "onnx-browser-webgpu": fake_encoder,
            "ollama-gguf-local": fake_tiny_chat,
        },
    )

    # Peek returns a tiny-chat-lane challenge — supervisor must route to
    # fake_tiny_chat, not fake_encoder.
    monkeypatch.setattr(
        "codepit_optimizer.orchestrator.peek_next_eligible_challenge",
        lambda _config: ("ch_gguf", "ollama-gguf-local"),
    )

    summary = run_optimizer_agent_forever(forever, install_signal_handlers=False)

    assert summary.iterations_completed == 1
    assert tiny_chat_calls == ["ch_gguf"]
    assert encoder_calls == []


def test_lane_runners_dispatch_to_encoder_when_eligible_challenge_is_onnx(
    tmp_path, monkeypatch
):
    """Symmetric to the tiny-chat case: an onnx-lane challenge routes to the
    encoder runner. Same registry; the supervisor's choice is purely a
    function of the peeked challenge's lane.
    """
    encoder_calls: list[str | None] = []
    tiny_chat_calls: list[str | None] = []

    def fake_encoder(config):
        encoder_calls.append(config.challenge_id)
        return _result(state="VERIFIED")

    def fake_tiny_chat(config):
        tiny_chat_calls.append(config.challenge_id)
        return _result(state="VERIFIED")

    forever = ForeverConfig(
        base_config=_base_config(tmp_path),
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
        max_iterations=1,
        lane_runners={
            "onnx-browser-webgpu": fake_encoder,
            "ollama-gguf-local": fake_tiny_chat,
        },
    )
    monkeypatch.setattr(
        "codepit_optimizer.orchestrator.peek_next_eligible_challenge",
        lambda _config: ("ch_enc", "onnx-browser-webgpu"),
    )

    summary = run_optimizer_agent_forever(forever, install_signal_handlers=False)

    assert summary.iterations_completed == 1
    assert encoder_calls == ["ch_enc"]
    assert tiny_chat_calls == []


def test_unknown_lane_is_skipped_not_fatal(tmp_path, monkeypatch):
    """If the network surfaces a challenge on a lane the agent has no runner
    for, the supervisor must skip the iteration (treat as no_challenge) and
    keep looping — not raise a fatal error. This is what lets an agent that
    only knows the encoder lane safely operate against a multi-lane network.
    """
    forever = ForeverConfig(
        base_config=_base_config(tmp_path),
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
        max_iterations=2,
        lane_runners={
            "onnx-browser-webgpu": lambda _c: _result(),
        },
    )
    monkeypatch.setattr(
        "codepit_optimizer.orchestrator.peek_next_eligible_challenge",
        lambda _config: ("ch_future", "future-image-gen-lane"),
    )

    summary = run_optimizer_agent_forever(forever, install_signal_handlers=False)

    assert summary.stopped_reason == "max_iterations"
    assert summary.iterations_started == 2
    assert summary.iterations_completed == 0
    assert summary.transient_error_count == 0


def test_default_lane_runners_registry_includes_both_lanes():
    """The default registry must out-of-the-box support both shipped lanes.
    This is what gives a freshly deployed agent (managed or external) the
    "join and pick whatever you're eligible for" property by default."""
    registry = default_lane_runners()
    assert set(registry.keys()) == {"onnx-browser-webgpu", "ollama-gguf-local"}


def test_lane_runners_none_preserves_legacy_direct_run_optimizer_agent_path(
    tmp_path, monkeypatch
):
    """Backward-compat: if a caller doesn't pass lane_runners, the supervisor
    keeps calling run_optimizer_agent directly (today's behavior). This is
    what keeps the existing forever-loop tests in this file working unchanged
    and gives operators a clean migration window before defaulting to dispatch.
    """
    calls: list[str | None] = []

    def fake_run(config):
        calls.append(config.challenge_id)
        return _result(state="VERIFIED")

    forever = ForeverConfig(
        base_config=_base_config(tmp_path),
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
        max_iterations=1,
        # lane_runners omitted — legacy path.
    )
    monkeypatch.setattr("codepit_optimizer.orchestrator.run_optimizer_agent", fake_run)

    summary = run_optimizer_agent_forever(forever, install_signal_handlers=False)

    assert summary.iterations_completed == 1
    assert calls == [None]  # legacy path leaves challenge_id unset (poll mode)


def test_loop_counts_terminal_states_across_iterations(tmp_path, monkeypatch):
    forever = ForeverConfig(
        base_config=_base_config(tmp_path),
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
        max_iterations=3,
    )
    sequence = iter([_result("VERIFIED"), _result("BENCHMARK_FAILED"), _result("VERIFIED")])
    monkeypatch.setattr(
        "codepit_optimizer.orchestrator.run_optimizer_agent",
        lambda _config: next(sequence),
    )

    summary = run_optimizer_agent_forever(forever, install_signal_handlers=False)

    assert summary.stopped_reason == "max_iterations"
    assert summary.iterations_started == 3
    assert summary.iterations_completed == 3
    assert summary.terminal_state_counts == {"VERIFIED": 2, "BENCHMARK_FAILED": 1}


def test_no_challenge_idles_without_treating_as_error(tmp_path, monkeypatch):
    forever = ForeverConfig(
        base_config=_base_config(tmp_path),
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
        max_iterations=2,
    )
    calls = {"count": 0}

    def fake_run(_config):
        calls["count"] += 1
        raise OrchestratorError("no eligible challenge returned for agent agt_x")

    monkeypatch.setattr("codepit_optimizer.orchestrator.run_optimizer_agent", fake_run)

    summary = run_optimizer_agent_forever(forever, install_signal_handlers=False)

    assert summary.stopped_reason == "max_iterations"
    assert summary.iterations_started == 2
    assert summary.iterations_completed == 0
    assert summary.transient_error_count == 0
    assert calls["count"] == 2


def test_transient_protocol_error_retries(tmp_path, monkeypatch):
    forever = ForeverConfig(
        base_config=_base_config(tmp_path),
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
        max_iterations=3,
    )

    sequence = [
        ProtocolError("rate limited", status_code=429, code="rate_limited", retryable=True),
        ProtocolError("rate limited", status_code=429, code="rate_limited", retryable=True),
        _result("VERIFIED"),
    ]
    queue = iter(sequence)

    def fake_run(_config):
        item = next(queue)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr("codepit_optimizer.orchestrator.run_optimizer_agent", fake_run)

    summary = run_optimizer_agent_forever(forever, install_signal_handlers=False)

    assert summary.stopped_reason == "max_iterations"
    assert summary.transient_error_count == 2
    assert summary.iterations_completed == 1
    assert summary.terminal_state_counts == {"VERIFIED": 1}


def test_non_retryable_protocol_error_stops_with_fatal_reason(tmp_path, monkeypatch):
    forever = ForeverConfig(
        base_config=_base_config(tmp_path),
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
        max_iterations=10,
    )
    monkeypatch.setattr(
        "codepit_optimizer.orchestrator.run_optimizer_agent",
        lambda _c: (_ for _ in ()).throw(
            ProtocolError(
                "credential revoked",
                status_code=401,
                code="auth.credential_revoked",
                retryable=False,
            ),
        ),
    )

    summary = run_optimizer_agent_forever(forever, install_signal_handlers=False)

    assert summary.stopped_reason == "fatal_error"
    assert summary.last_error and "credential revoked" in summary.last_error


def test_stop_event_breaks_between_iterations(tmp_path, monkeypatch):
    forever = ForeverConfig(
        base_config=_base_config(tmp_path),
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
    )
    stop_event = threading.Event()
    invocations = {"count": 0}

    def fake_run(_config):
        invocations["count"] += 1
        # After the first iteration, request shutdown — the loop must
        # observe the event and exit before invoking us again.
        stop_event.set()
        return _result("VERIFIED")

    monkeypatch.setattr("codepit_optimizer.orchestrator.run_optimizer_agent", fake_run)

    summary = run_optimizer_agent_forever(
        forever,
        stop_event=stop_event,
        install_signal_handlers=False,
    )

    assert summary.stopped_reason == "stop_event"
    assert invocations["count"] == 1
    assert summary.iterations_completed == 1


def test_on_iteration_callback_failures_do_not_break_the_loop(tmp_path, monkeypatch):
    captured: list[tuple[int, str]] = []

    def callback(iteration: int, outcome: ForeverIterationOutcome) -> None:
        captured.append((iteration, outcome.kind))
        raise RuntimeError("callback intentionally raises")

    forever = ForeverConfig(
        base_config=_base_config(tmp_path),
        idle_sleep_s=0.0,
        error_backoff_s=0.0,
        max_iterations=2,
        on_iteration=callback,
    )

    monkeypatch.setattr(
        "codepit_optimizer.orchestrator.run_optimizer_agent",
        lambda _c: _result("VERIFIED"),
    )

    summary = run_optimizer_agent_forever(forever, install_signal_handlers=False)

    assert summary.iterations_completed == 2
    assert captured == [(1, "result"), (2, "result")]
