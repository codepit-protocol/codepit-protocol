import json
from pathlib import Path

import httpx

from codepit_optimizer.managed_worker import (
    ManagedRunContext,
    ManagedWorkerService,
    ManagedWorkerRunRequest,
    run_managed_worker_once,
)
from codepit_optimizer.orchestrator import (
    OrchestratorConfig,
    OrchestratorError,
    OrchestratorResult,
    TinyChatRunConfig,
    run_tiny_chat_lane,
)
from codepit_optimizer.tiny_chat_bundle import OLLAMA_GGUF_LOCAL_ARTIFACT_LANE


def _request() -> ManagedWorkerRunRequest:
    return ManagedWorkerRunRequest(
        agent_id="agent_managed_1",
        runtime_credential="runtime-secret",
        engine_base_url="http://engine.test",
        managed_run=ManagedRunContext(
            runtime_id="rt_managed_1",
            agent_id="agent_managed_1",
            challenge_id="ch_tiny_chat_1",
            artifact_lane=OLLAMA_GGUF_LOCAL_ARTIFACT_LANE,
            runtime_family="chromium",
            worker_pool="railway-cpu",
            lease_owner="tick:stage-c:rt_managed_1",
            lease_expires_at="2026-05-23T00:06:00.000Z",
            per_run_wall_clock_limit_ms=120_000,
            per_run_attempt_limit=1,
            bootstrap_eligible=True,
        ),
    )


def test_managed_worker_completes_success_after_submission(tmp_path: Path) -> None:
    captured_configs: list[OrchestratorConfig] = []
    completion_requests: list[httpx.Request] = []

    def runner(config: OrchestratorConfig) -> OrchestratorResult:
        captured_configs.append(config)
        return OrchestratorResult(
            agent_id="agent_managed_1",
            signer_address="0x0000000000000000000000000000000000000001",
            challenge_id="ch_tiny_chat_1",
            submission_id="sub_success_1",
            state="VERIFIED",
            benchmark_target_version="tiny-chat-v1",
            chosen_recipe="tiny-chat-smoke",
            bundle_dir=tmp_path / "bundle",
        )

    def handler(request: httpx.Request) -> httpx.Response:
        completion_requests.append(request)
        return httpx.Response(200, json={"runtime_id": "rt_managed_1", "status": "IDLE"})

    result = run_managed_worker_once(
        _request(),
        work_root=tmp_path,
        lane_runners={OLLAMA_GGUF_LOCAL_ARTIFACT_LANE: runner},
        transport=httpx.MockTransport(handler),
    )

    assert result.outcome == "success"
    assert result.submission_id == "sub_success_1"
    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert config.agent_id == "agent_managed_1"
    assert config.runtime_credential == "runtime-secret"
    assert config.challenge_id == "ch_tiny_chat_1"
    assert config.client_submission_id == "managed-rt_managed_1-ch_tiny_chat_1"
    assert config.poll_timeout_s == 120.0
    assert config.request_timeout_s == 120.0

    assert len(completion_requests) == 1
    completion = completion_requests[0]
    assert completion.method == "POST"
    assert completion.url == httpx.URL("http://engine.test/v2/managed-runs/rt_managed_1/complete")
    assert completion.headers["authorization"] == "Bearer runtime-secret"
    assert json.loads(completion.content) == {
        "outcome": "success",
        "submission_id": "sub_success_1",
    }


def test_managed_worker_wires_engine_routed_brain_from_run_request(tmp_path: Path) -> None:
    captured_configs: list[OrchestratorConfig] = []

    def runner(config: OrchestratorConfig) -> OrchestratorResult:
        captured_configs.append(config)
        return OrchestratorResult(
            agent_id=config.agent_id or "",
            signer_address="0x0000000000000000000000000000000000000001",
            challenge_id=config.challenge_id or "",
            submission_id="sub_brain_1",
            state="VERIFIED",
            benchmark_target_version="tiny-chat-v1",
            chosen_recipe="q4_k_m",
            bundle_dir=tmp_path / "bundle",
        )

    request = ManagedWorkerRunRequest(
        **{
            **_request().__dict__,
            "brain_config": {
                "enabled": True,
                "provider": "managed",
                "tier": "premium",
                "timeout_s": 45,
                "require_llm_brain": True,
            },
        }
    )

    result = run_managed_worker_once(
        request,
        work_root=tmp_path,
        lane_runners={OLLAMA_GGUF_LOCAL_ARTIFACT_LANE: runner},
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, json={"runtime_id": "rt_managed_1", "status": "IDLE"})
        ),
    )

    assert result.outcome == "success"
    assert len(captured_configs) == 1
    brain = captured_configs[0].brain
    assert brain is not None
    assert brain.config.provider_name == "managed"
    assert brain.config.tier == "premium"
    assert brain.config.fallback_on_error is False


def test_managed_worker_retries_success_completion_after_transient_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    completion_requests: list[httpx.Request] = []

    def runner(config: OrchestratorConfig) -> OrchestratorResult:
        return OrchestratorResult(
            agent_id=config.agent_id or "",
            signer_address="0x0000000000000000000000000000000000000001",
            challenge_id=config.challenge_id or "",
            submission_id="sub_success_retry_1",
            state="VERIFIED",
            benchmark_target_version="tiny-chat-v1",
            chosen_recipe="tiny-chat-smoke",
            bundle_dir=tmp_path / "bundle",
        )

    def handler(request: httpx.Request) -> httpx.Response:
        completion_requests.append(request)
        if len(completion_requests) == 1:
            raise httpx.ReadTimeout("engine verifier is busy")
        return httpx.Response(200, json={"runtime_id": "rt_managed_1", "status": "IDLE"})

    monkeypatch.setattr("codepit_optimizer.managed_worker.time.sleep", lambda _seconds: None)

    result = run_managed_worker_once(
        _request(),
        work_root=tmp_path,
        lane_runners={OLLAMA_GGUF_LOCAL_ARTIFACT_LANE: runner},
        transport=httpx.MockTransport(handler),
    )

    assert result.outcome == "success"
    assert result.submission_id == "sub_success_retry_1"
    assert len(completion_requests) == 2
    assert [request.url.path for request in completion_requests] == [
        "/v2/managed-runs/rt_managed_1/complete",
        "/v2/managed-runs/rt_managed_1/complete",
    ]


def test_managed_worker_default_tiny_chat_runner_adapts_config(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_configs: list[TinyChatRunConfig] = []
    completion_requests: list[httpx.Request] = []

    def runner(config: TinyChatRunConfig) -> OrchestratorResult:
        captured_configs.append(config)
        assert isinstance(config, TinyChatRunConfig)
        return OrchestratorResult(
            agent_id=config.agent_id or "",
            signer_address="0x0000000000000000000000000000000000000001",
            challenge_id=config.challenge_id or "",
            submission_id="sub_default_tiny_chat_1",
            state="VERIFIED",
            benchmark_target_version="tiny-chat-v1",
            chosen_recipe="q4_k_m",
            bundle_dir=tmp_path / "bundle",
        )

    def handler(request: httpx.Request) -> httpx.Response:
        completion_requests.append(request)
        return httpx.Response(200, json={"runtime_id": "rt_managed_1", "status": "IDLE"})

    for key in (
        "CODEPIT_TINY_CHAT_GGUF_PATH",
        "CODEPIT_TINY_CHAT_GGUF_URL",
        "CODEPIT_TINY_CHAT_GGUF_CACHE_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(
        "codepit_optimizer.orchestrator.run_tiny_chat_external_agent",
        runner,
    )

    result = run_managed_worker_once(
        _request(),
        work_root=tmp_path,
        transport=httpx.MockTransport(handler),
    )

    assert result.outcome == "success"
    assert result.submission_id == "sub_default_tiny_chat_1"
    assert len(captured_configs) == 1
    config = captured_configs[0]
    assert config.base_url == "http://engine.test"
    assert config.agent_id == "agent_managed_1"
    assert config.runtime_credential == "runtime-secret"
    assert config.challenge_id == "ch_tiny_chat_1"
    assert config.client_submission_id == "managed-rt_managed_1-ch_tiny_chat_1"
    assert config.poll_timeout_s == 120.0
    assert config.receipt_poll_timeout_s == 120.0
    assert config.request_timeout_s == 120.0
    assert len(completion_requests) == 1


def test_tiny_chat_lane_downloads_env_gguf_url(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cache_path = tmp_path / "cache" / "tiny-chat.gguf"
    captured_configs: list[TinyChatRunConfig] = []
    requested_urls: list[str] = []

    class FakeResponse:
        def __init__(self) -> None:
            self._chunks = [b"GG", b"UFpayload", b""]

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def read(self, _size: int) -> bytes:
            return self._chunks.pop(0)

    def fake_urlopen(request, timeout: int):
        requested_urls.append(request.full_url)
        assert timeout == 60
        return FakeResponse()

    def runner(config: TinyChatRunConfig) -> OrchestratorResult:
        captured_configs.append(config)
        return OrchestratorResult(
            agent_id=config.agent_id or "",
            signer_address="0x0000000000000000000000000000000000000001",
            challenge_id=config.challenge_id or "",
            submission_id="sub_url_gguf_1",
            state="VERIFIED",
            benchmark_target_version="tiny-chat-v1",
            chosen_recipe="q4_k_m",
            bundle_dir=tmp_path / "bundle",
        )

    monkeypatch.setenv("CODEPIT_TINY_CHAT_GGUF_URL", "https://example.test/model.gguf")
    monkeypatch.setenv("CODEPIT_TINY_CHAT_GGUF_CACHE_PATH", str(cache_path))
    monkeypatch.delenv("CODEPIT_TINY_CHAT_GGUF_PATH", raising=False)
    monkeypatch.setattr("codepit_optimizer.orchestrator.urlopen", fake_urlopen)
    monkeypatch.setattr(
        "codepit_optimizer.orchestrator.run_tiny_chat_external_agent",
        runner,
    )

    result = run_tiny_chat_lane(
        OrchestratorConfig(
            base_url="http://engine.test",
            work_dir=tmp_path / "work",
            agent_id="agent_managed_1",
            runtime_credential="runtime-secret",
            challenge_id="ch_tiny_chat_1",
        )
    )

    assert result.submission_id == "sub_url_gguf_1"
    assert requested_urls == ["https://example.test/model.gguf"]
    assert cache_path.read_bytes() == b"GGUFpayload"
    assert captured_configs[0].gguf_path == cache_path


def test_managed_worker_completes_failure_with_stable_error_code(tmp_path: Path) -> None:
    completion_requests: list[httpx.Request] = []

    def runner(_config: OrchestratorConfig) -> OrchestratorResult:
        raise OrchestratorError("tiny chat packaging failed")

    def handler(request: httpx.Request) -> httpx.Response:
        completion_requests.append(request)
        return httpx.Response(200, json={"runtime_id": "rt_managed_1", "status": "ERROR"})

    result = run_managed_worker_once(
        _request(),
        work_root=tmp_path,
        lane_runners={OLLAMA_GGUF_LOCAL_ARTIFACT_LANE: runner},
        transport=httpx.MockTransport(handler),
    )

    assert result.outcome == "failure"
    assert result.error_code == "worker.orchestrator_error"
    assert len(completion_requests) == 1
    assert json.loads(completion_requests[0].content) == {
        "outcome": "failure",
        "error_code": "worker.orchestrator_error",
    }


def test_managed_worker_service_accepts_authenticated_run_and_rejects_missing_auth(tmp_path: Path) -> None:
    completion_requests: list[httpx.Request] = []

    def runner(config: OrchestratorConfig) -> OrchestratorResult:
        return OrchestratorResult(
            agent_id=config.agent_id or "",
            signer_address="0x0000000000000000000000000000000000000001",
            challenge_id=config.challenge_id or "",
            submission_id="sub_service_1",
            state="VERIFIED",
            benchmark_target_version="tiny-chat-v1",
            chosen_recipe="tiny-chat-smoke",
            bundle_dir=tmp_path / "bundle",
        )

    def handler(request: httpx.Request) -> httpx.Response:
        completion_requests.append(request)
        return httpx.Response(200, json={"runtime_id": "rt_managed_1", "status": "IDLE"})

    service = ManagedWorkerService(
        shared_secret="worker-shared-secret",
        work_root=tmp_path,
        lane_runners={OLLAMA_GGUF_LOCAL_ARTIFACT_LANE: runner},
        transport=httpx.MockTransport(handler),
        run_in_background=False,
    )

    response = service.accept_run(
        {
            "agent_id": "agent_managed_1",
            "runtime_credential": "runtime-secret",
            "engine_base_url": "http://engine.test",
            "brain_config": None,
            "managed_run": _request().managed_run.__dict__,
        },
        authorization="Bearer worker-shared-secret",
    )

    assert response["worker_id"] == "railway-cpu:rt_managed_1"
    assert response["runtime_ref"] == "http-run:rt_managed_1"
    assert len(completion_requests) == 1

    try:
        service.accept_run({}, authorization=None)
        raise AssertionError("expected auth failure")
    except PermissionError as error:
        assert "unauthorized" in str(error)


def test_managed_worker_service_rejects_agent_id_mismatch(tmp_path: Path) -> None:
    service = ManagedWorkerService(
        shared_secret="worker-shared-secret",
        work_root=tmp_path,
        run_in_background=False,
    )
    body = {
        "agent_id": "agent_top_level",
        "runtime_credential": "runtime-secret",
        "engine_base_url": "http://engine.test",
        "managed_run": {**_request().managed_run.__dict__, "agent_id": "agent_inside_run"},
    }

    try:
        service.accept_run(body, authorization="Bearer worker-shared-secret")
        raise AssertionError("expected bad request")
    except ValueError as error:
        assert "agent_id mismatch" in str(error)
