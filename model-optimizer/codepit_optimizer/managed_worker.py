"""Bounded managed-worker runner for Railway CPU deployments.

The engine dispatches one claimed managed-runtime lease to this worker. The
worker runs exactly one lane-specific optimizer path, then calls the engine's
managed-run completion endpoint with the same per-run bearer credential.
"""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import httpx

from .brain import Brain, BrainConfig
from .brain_providers import ManagedBrainProvider
from .orchestrator import (
    LaneRunner,
    OrchestratorConfig,
    OrchestratorError,
    OrchestratorResult,
    default_lane_runners,
)
from .protocol import CodePitClient, ProtocolError

COMPLETION_RETRY_ATTEMPTS = 6
COMPLETION_RETRY_BACKOFF_S = 5.0
COMPLETION_TIMEOUT_FLOOR_S = 30.0
COMPLETION_TIMEOUT_CEILING_S = 180.0
MANAGED_WORKER_BRAIN_ENABLED_ENV = "CODEPIT_MANAGED_WORKER_BRAIN_ENABLED"
MANAGED_WORKER_BRAIN_TIER_ENV = "CODEPIT_MANAGED_WORKER_BRAIN_TIER"
MANAGED_WORKER_BRAIN_TIMEOUT_ENV = "CODEPIT_MANAGED_WORKER_BRAIN_TIMEOUT_SECONDS"
MANAGED_WORKER_REQUIRE_LLM_BRAIN_ENV = "CODEPIT_MANAGED_WORKER_REQUIRE_LLM_BRAIN"


@dataclass(frozen=True)
class ManagedRunContext:
    runtime_id: str
    agent_id: str
    challenge_id: str
    artifact_lane: str
    runtime_family: str
    worker_pool: str
    lease_owner: str
    lease_expires_at: str
    per_run_wall_clock_limit_ms: int
    per_run_attempt_limit: int
    bootstrap_eligible: bool


@dataclass(frozen=True)
class ManagedWorkerRunRequest:
    agent_id: str
    runtime_credential: str
    engine_base_url: str
    managed_run: ManagedRunContext
    brain_config: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class ManagedWorkerRunResult:
    outcome: str
    runtime_id: str
    submission_id: str | None = None
    error_code: str | None = None
    completion_response: Mapping[str, Any] | None = None


class ManagedWorkerBadRequest(ValueError):
    """Raised when the engine -> worker handoff payload is malformed."""


class ManagedWorkerService:
    def __init__(
        self,
        *,
        shared_secret: str,
        work_root: Path,
        lane_runners: Mapping[str, LaneRunner] | None = None,
        transport: httpx.BaseTransport | None = None,
        run_in_background: bool = True,
    ) -> None:
        if not shared_secret:
            raise ValueError("shared_secret is required")
        self._shared_secret = shared_secret
        self._work_root = work_root
        self._lane_runners = lane_runners
        self._transport = transport
        self._run_in_background = run_in_background

    def accept_run(
        self,
        body: Mapping[str, Any],
        *,
        authorization: str | None,
    ) -> dict[str, str]:
        if authorization != f"Bearer {self._shared_secret}":
            raise PermissionError("unauthorized managed-worker request")

        request = parse_managed_worker_request(body)
        if self._run_in_background:
            thread = threading.Thread(
                target=self._run_safely,
                args=(request,),
                name=f"codepit-managed-worker-{request.managed_run.runtime_id}",
                daemon=True,
            )
            thread.start()
        else:
            self._run_safely(request)

        return {
            "worker_id": f"railway-cpu:{request.managed_run.runtime_id}",
            "runtime_ref": f"http-run:{request.managed_run.runtime_id}",
        }

    def _run_safely(self, request: ManagedWorkerRunRequest) -> None:
        run_managed_worker_once(
            request,
            work_root=self._work_root,
            lane_runners=self._lane_runners,
            transport=self._transport,
        )


def parse_managed_worker_request(body: Mapping[str, Any]) -> ManagedWorkerRunRequest:
    managed_run_raw = body.get("managed_run")
    if not isinstance(managed_run_raw, Mapping):
        raise ManagedWorkerBadRequest("managed_run is required")
    agent_id = _required_string(body, "agent_id")
    managed_run_agent_id = _required_string(managed_run_raw, "agent_id")
    if managed_run_agent_id != agent_id:
        raise ManagedWorkerBadRequest("agent_id mismatch between request and managed_run")

    return ManagedWorkerRunRequest(
        agent_id=agent_id,
        runtime_credential=_required_string(body, "runtime_credential"),
        engine_base_url=_required_string(body, "engine_base_url"),
        brain_config=body.get("brain_config") if isinstance(body.get("brain_config"), Mapping) else None,
        managed_run=ManagedRunContext(
            runtime_id=_required_string(managed_run_raw, "runtime_id"),
            agent_id=managed_run_agent_id,
            challenge_id=_required_string(managed_run_raw, "challenge_id"),
            artifact_lane=_required_string(managed_run_raw, "artifact_lane"),
            runtime_family=_required_string(managed_run_raw, "runtime_family"),
            worker_pool=_required_string(managed_run_raw, "worker_pool"),
            lease_owner=_required_string(managed_run_raw, "lease_owner"),
            lease_expires_at=_required_string(managed_run_raw, "lease_expires_at"),
            per_run_wall_clock_limit_ms=_required_int(managed_run_raw, "per_run_wall_clock_limit_ms"),
            per_run_attempt_limit=_required_int(managed_run_raw, "per_run_attempt_limit"),
            bootstrap_eligible=bool(managed_run_raw.get("bootstrap_eligible")),
        ),
    )


def run_managed_worker_http_server(
    *,
    host: str = "0.0.0.0",
    port: int = 8080,
    shared_secret: str,
    work_root: Path,
) -> None:
    service = ManagedWorkerService(shared_secret=shared_secret, work_root=work_root)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - stdlib API name
            if self.path == "/health":
                _write_json(self, 200, {"ok": True, "service": "codepit-managed-worker"})
                return
            _write_json(self, 404, {"error": "not_found"})

        def do_POST(self) -> None:  # noqa: N802 - stdlib API name
            if self.path != "/v2/managed-worker/runs":
                _write_json(self, 404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("content-length", "0"))
                raw = self.rfile.read(length)
                body = json.loads(raw.decode("utf-8")) if raw else {}
                if not isinstance(body, Mapping):
                    raise ManagedWorkerBadRequest("JSON object body is required")
                response = service.accept_run(
                    body,
                    authorization=self.headers.get("authorization"),
                )
                _write_json(self, 202, response)
            except PermissionError:
                _write_json(self, 401, {"error": "unauthorized"})
            except (ManagedWorkerBadRequest, json.JSONDecodeError, UnicodeDecodeError) as error:
                _write_json(self, 400, {"error": str(error)})

        def log_message(self, _format: str, *_args: object) -> None:
            return

    ThreadingHTTPServer((host, port), Handler).serve_forever()


def run_managed_worker_once(
    request: ManagedWorkerRunRequest,
    *,
    work_root: Path,
    lane_runners: Mapping[str, LaneRunner] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> ManagedWorkerRunResult:
    """Execute one managed-run handoff and release the engine lease.

    The worker reports success only after the lane runner returns a submission
    id. Terminal runner failures are converted to stable engine-facing error
    codes and still release the lease via the completion endpoint.
    """

    managed_run = request.managed_run
    runners = lane_runners or default_lane_runners()
    runner = runners.get(managed_run.artifact_lane)
    if runner is None:
        return _complete_failure(
            request,
            error_code="worker.unsupported_artifact_lane",
            transport=transport,
        )

    config = OrchestratorConfig(
        base_url=request.engine_base_url,
        work_dir=work_root / managed_run.runtime_id,
        agent_id=request.agent_id,
        runtime_credential=request.runtime_credential,
        challenge_id=managed_run.challenge_id,
        poll_timeout_s=max(managed_run.per_run_wall_clock_limit_ms / 1000.0, 1.0),
        receipt_poll_timeout_s=max(managed_run.per_run_wall_clock_limit_ms / 1000.0, 1.0),
        request_timeout_s=_request_timeout_s(managed_run),
        session_path=None,
        client_submission_id=_client_submission_id(managed_run),
        brain=_build_managed_brain(request, runtime_id=managed_run.runtime_id),
    )

    try:
        result = runner(config)
    except OrchestratorError:
        return _complete_failure(
            request,
            error_code="worker.orchestrator_error",
            transport=transport,
        )
    except ProtocolError:
        return _complete_failure(
            request,
            error_code="worker.protocol_error",
            transport=transport,
        )
    except Exception:
        return _complete_failure(
            request,
            error_code="worker.unhandled_error",
            transport=transport,
        )

    return _complete_success(request, result, transport=transport)


def _complete_success(
    request: ManagedWorkerRunRequest,
    result: OrchestratorResult,
    *,
    transport: httpx.BaseTransport | None,
) -> ManagedWorkerRunResult:
    body = {"outcome": "success", "submission_id": result.submission_id}
    response = _complete_managed_run_with_retry(
        request,
        request.managed_run.runtime_id,
        body,
        transport=transport,
    )
    return ManagedWorkerRunResult(
        outcome="success",
        runtime_id=request.managed_run.runtime_id,
        submission_id=result.submission_id,
        completion_response=response,
    )


def _complete_failure(
    request: ManagedWorkerRunRequest,
    *,
    error_code: str,
    transport: httpx.BaseTransport | None,
) -> ManagedWorkerRunResult:
    response = _complete_managed_run_with_retry(
        request,
        request.managed_run.runtime_id,
        {"outcome": "failure", "error_code": error_code},
        transport=transport,
    )
    return ManagedWorkerRunResult(
        outcome="failure",
        runtime_id=request.managed_run.runtime_id,
        error_code=error_code,
        completion_response=response,
    )


def _completion_client(
    request: ManagedWorkerRunRequest,
    *,
    transport: httpx.BaseTransport | None,
) -> CodePitClient:
    return CodePitClient(
        request.engine_base_url,
        agent_id=request.agent_id,
        credential=request.runtime_credential,
        transport=transport,
        timeout=_completion_timeout_s(request),
    )


def _complete_managed_run_with_retry(
    request: ManagedWorkerRunRequest,
    runtime_id: str,
    body: Mapping[str, Any],
    *,
    transport: httpx.BaseTransport | None,
) -> dict[str, Any]:
    last_error: ProtocolError | None = None
    for attempt in range(COMPLETION_RETRY_ATTEMPTS):
        try:
            return _completion_client(request, transport=transport).complete_managed_run(runtime_id, body)
        except ProtocolError as error:
            if error.retryable is not True or attempt == COMPLETION_RETRY_ATTEMPTS - 1:
                raise
            last_error = error
            time.sleep(COMPLETION_RETRY_BACKOFF_S)

    assert last_error is not None
    raise last_error


def _completion_timeout_s(request: ManagedWorkerRunRequest) -> float:
    wall_clock_s = request.managed_run.per_run_wall_clock_limit_ms / 1000.0
    return min(
        max(wall_clock_s, COMPLETION_TIMEOUT_FLOOR_S),
        COMPLETION_TIMEOUT_CEILING_S,
    )


def _request_timeout_s(managed_run: ManagedRunContext) -> float:
    return min(
        max(managed_run.per_run_wall_clock_limit_ms / 1000.0, COMPLETION_TIMEOUT_FLOOR_S),
        COMPLETION_TIMEOUT_CEILING_S,
    )


def _client_submission_id(managed_run: ManagedRunContext) -> str:
    return f"managed-{managed_run.runtime_id}-{managed_run.challenge_id}"[:128]


def _build_managed_brain(
    request: ManagedWorkerRunRequest,
    *,
    runtime_id: str,
) -> Brain | None:
    brain_config = request.brain_config or {}
    if not _brain_enabled(brain_config):
        return None
    provider_name = str(brain_config.get("provider") or "managed").strip().lower()
    if provider_name != "managed":
        raise ValueError("managed worker only supports engine-routed managed brain")
    tier = str(
        brain_config.get("tier")
        or os.environ.get(MANAGED_WORKER_BRAIN_TIER_ENV)
        or "cheap"
    )
    provider = ManagedBrainProvider(
        base_url=request.engine_base_url,
        bearer_token=request.runtime_credential,
        timeout_s=_brain_timeout_seconds(brain_config),
    )
    return Brain(
        config=BrainConfig(
            provider_name="managed",
            tier=tier,
            fallback_on_error=not _brain_required(brain_config),
            action_id_prefix=f"managed-{runtime_id}",
        ),
        provider=provider,
    )


def _brain_enabled(brain_config: Mapping[str, Any]) -> bool:
    if "enabled" in brain_config:
        return bool(brain_config["enabled"])
    return _env_bool(MANAGED_WORKER_BRAIN_ENABLED_ENV, True)


def _brain_required(brain_config: Mapping[str, Any]) -> bool:
    for key in ("require_llm_brain", "strict"):
        if key in brain_config:
            return bool(brain_config[key])
    return _env_bool(MANAGED_WORKER_REQUIRE_LLM_BRAIN_ENV, True)


def _brain_timeout_seconds(brain_config: Mapping[str, Any]) -> float:
    raw = (
        brain_config.get("timeout_s")
        or brain_config.get("timeout_seconds")
        or os.environ.get(MANAGED_WORKER_BRAIN_TIMEOUT_ENV)
        or 60.0
    )
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as error:
        raise ValueError("brain timeout must be numeric") from error
    return min(max(timeout, 1.0), COMPLETION_TIMEOUT_CEILING_S)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _required_string(body: Mapping[str, Any], key: str) -> str:
    value = body.get(key)
    if not isinstance(value, str) or not value:
        raise ManagedWorkerBadRequest(f"{key} is required")
    return value


def _required_int(body: Mapping[str, Any], key: str) -> int:
    value = body.get(key)
    if not isinstance(value, int):
        raise ManagedWorkerBadRequest(f"{key} is required")
    return value


def _write_json(handler: BaseHTTPRequestHandler, status: int, body: Mapping[str, Any]) -> None:
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def main() -> None:
    shared_secret = os.environ.get("V2_MANAGED_WORKER_SHARED_SECRET", "")
    if not shared_secret:
        raise SystemExit("V2_MANAGED_WORKER_SHARED_SECRET is required")
    port = int(os.environ.get("PORT", "8080"))
    work_root = Path(os.environ.get("CODEPIT_MANAGED_WORKER_WORK_DIR", "/tmp/codepit-managed-worker"))
    run_managed_worker_http_server(
        port=port,
        shared_secret=shared_secret,
        work_root=work_root,
    )


if __name__ == "__main__":
    main()
