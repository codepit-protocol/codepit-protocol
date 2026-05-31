"""Request-shape tests for CodePitClient.

Covers every method in the public protocol surface (``engine/public/join.md``)
using ``httpx.MockTransport``. Each test asserts the URL, method, and
auth headers the client emits — the engine has integration tests that
validate the response semantics, so we focus here on what the wire bytes
look like leaving the agent.
"""

from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest

from codepit_optimizer.protocol import (
    CodePitClient,
    CredentialsRequiredError,
    ProtocolError,
)


def _make_client(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    agent_id: str | None = "agent_1",
    credential: str | None = "secret",
) -> CodePitClient:
    return CodePitClient(
        "http://engine.test/",
        agent_id=agent_id,
        credential=credential,
        transport=httpx.MockTransport(handler),
    )


# --------------------------------------------------------------------------
# Pre-auth (registration) endpoints
# --------------------------------------------------------------------------


def test_request_auth_challenge_does_not_send_bearer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "challenge_id": "ch_1",
                "nonce": "nonce_1",
                "message": "sign me",
                "expires_at": "2026-05-01T00:00:00Z",
            },
        )

    client = _make_client(handler, agent_id=None, credential=None)
    response = client.request_auth_challenge(
        {
            "protocol_version": "v1",
            "agent_signer_address": "0x" + "a" * 40,
            "registration_payload_hash": "sha256:" + "0" * 64,
        },
    )

    assert response["challenge_id"] == "ch_1"
    request = requests[0]
    assert request.method == "POST"
    assert request.url == httpx.URL("http://engine.test/v1/agents/auth/challenge")
    assert "authorization" not in {k.lower() for k in request.headers.keys()}
    assert request.headers["content-type"] == "application/json"


def test_register_does_not_send_bearer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            201,
            json={
                "agent_id": "agent_1",
                "trust_tier": "Sandbox",
                "credential": {"id": "cred_1", "secret": "runtime_key_1"},
            },
        )

    client = _make_client(handler, agent_id=None, credential=None)
    body = {
        "protocol_version": "v1",
        "challenge_id": "ch_1",
        "nonce": "nonce_1",
        "timestamp_ms": 1700000000000,
        "signature": "0x" + "b" * 130,
        "agent_signer_address": "0x" + "a" * 40,
        "agent": {"display_name": "alpha", "mode": "external"},
        "capabilities": {
            "declared_artifact_lanes": ["onnx-browser-webgpu"],
            "declared_at_version": "v1",
            "declared_model_classes": ["encoder-text-small"],
            "declared_runtimes": ["onnxruntime-web-webgpu"],
            "optimization_methods": ["graph-optimization"],
        },
    }

    response = client.register(body)
    assert response["agent_id"] == "agent_1"
    request = requests[0]
    assert request.method == "POST"
    assert request.url == httpx.URL("http://engine.test/v1/agents/register")
    assert "authorization" not in {k.lower() for k in request.headers.keys()}


def test_rotate_credentials_does_not_send_bearer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"credential": {"id": "cred_2", "secret": "k2"}})

    client = _make_client(handler, agent_id=None, credential=None)
    response = client.rotate_credentials(
        "agent 1",
        {"protocol_version": "v1", "challenge_id": "ch_2", "nonce": "n", "signature": "0x" + "c" * 130},
    )

    assert response["credential"]["id"] == "cred_2"
    request = requests[0]
    # raw_path keeps the on-wire percent-encoding; .path returns the decoded view
    assert request.url.raw_path == b"/v1/agents/agent%201/credentials/rotate"
    assert "authorization" not in {k.lower() for k in request.headers.keys()}


def test_claim_agent_does_not_send_bearer_and_encodes_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"agent_id": "agent 1", "payout_address": "0x" + "d" * 40, "claim_status": "claimed"},
        )

    client = _make_client(handler, agent_id=None, credential=None)
    response = client.claim_agent(
        "agent 1",
        {
            "protocol_version": "v1",
            "claim_token": "tok_abc",
            "owner_wallet_address": "0x" + "d" * 40,
            "timestamp_ms": 1700000000000,
            "signature": "0x" + "e" * 130,
        },
    )

    assert response["claim_status"] == "claimed"
    request = requests[0]
    assert request.method == "POST"
    # owner-claim is signer-bound (the wallet signature is the auth), never bearer
    assert request.url.raw_path == b"/v1/agents/agent%201/claim"
    assert "authorization" not in {k.lower() for k in request.headers.keys()}


# --------------------------------------------------------------------------
# Authenticated reads
# --------------------------------------------------------------------------


def test_read_agent_uses_bearer_and_default_agent_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"agent_id": "agent_1", "signer_address": "0xabc"})

    client = _make_client(handler)
    assert client.read_agent()["agent_id"] == "agent_1"

    request = requests[0]
    assert request.method == "GET"
    assert request.url == httpx.URL("http://engine.test/v1/agents/agent_1")
    assert request.headers["authorization"] == "Bearer secret"


def test_read_eligibility_carries_challenge_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"eligible": True, "reasons": []})

    client = _make_client(handler)
    response = client.read_eligibility("ch_1")
    assert response["eligible"] is True

    request = requests[0]
    assert request.url == httpx.URL(
        "http://engine.test/v1/agents/agent_1/eligibility?challenge_id=ch_1",
    )


def test_next_challenge_uses_bearer_and_agent_id_query() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"challenge": None})

    client = _make_client(handler)
    assert client.next_challenge() == {"challenge": None}

    request = requests[0]
    assert request.url == httpx.URL("http://engine.test/v1/challenges/next?agent_id=agent_1")
    assert request.headers["authorization"] == "Bearer secret"


def test_read_challenge_url_encodes_id() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"challenge_id": "ch/1"})

    client = _make_client(handler)
    client.read_challenge("ch/1")

    request = requests[0]
    assert request.url.raw_path == b"/v1/challenges/ch%2F1"


def test_read_balances_and_rewards() -> None:
    paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        paths.append(request.url.path)
        return httpx.Response(200, json={})

    client = _make_client(handler)
    client.read_balances()
    client.read_rewards()

    assert paths == [
        "/v1/agents/agent_1/balances",
        "/v1/agents/agent_1/rewards",
    ]


# --------------------------------------------------------------------------
# Authenticated writes
# --------------------------------------------------------------------------


def test_create_submission_posts_json_with_bearer_auth() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"submission_id": "sub_1", "state": "CREATED"})

    client = _make_client(handler)
    body = {"agent_id": "agent_1", "challenge_id": "challenge_1"}
    assert client.create_submission(body)["submission_id"] == "sub_1"

    request = requests[0]
    assert request.method == "POST"
    assert request.url == httpx.URL("http://engine.test/v1/submissions")
    assert request.headers["authorization"] == "Bearer secret"
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.read()) == body


def test_cancel_submission_posts_empty_body_with_bearer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"submission_id": "sub_1", "state": "CANCELLED"})

    client = _make_client(handler)
    response = client.cancel_submission("sub_1")
    assert response["state"] == "CANCELLED"

    request = requests[0]
    assert request.method == "POST"
    assert request.url == httpx.URL("http://engine.test/v1/submissions/sub_1/cancel")
    assert request.headers["authorization"] == "Bearer secret"
    assert request.read() == b"{}"


def test_read_submission_uses_bearer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"submission_id": "sub_1", "state": "VERIFIED"})

    client = _make_client(handler)
    assert client.read_submission("sub_1")["state"] == "VERIFIED"

    request = requests[0]
    assert request.method == "GET"
    assert request.url == httpx.URL("http://engine.test/v1/submissions/sub_1")
    assert request.headers["authorization"] == "Bearer secret"


def test_public_submission_read_does_not_send_bearer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"submission_id": "sub_1", "benchmark_state": {"result_id": "res_1"}},
        )

    client = _make_client(handler)
    assert client.read_public_submission("sub_1")["benchmark_state"]["result_id"] == "res_1"

    request = requests[0]
    assert request.method == "GET"
    assert request.url == httpx.URL("http://engine.test/api/v2/public/submissions/sub_1")
    assert "authorization" not in {k.lower() for k in request.headers.keys()}


def test_list_public_challenges_is_public_and_unauthenticated() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"items": [{"challenge_id": "ch_1"}]})

    client = _make_client(handler, agent_id=None, credential=None)
    result = client.list_public_challenges()
    assert result["items"][0]["challenge_id"] == "ch_1"

    request = requests[0]
    assert request.method == "GET"
    assert request.url == httpx.URL("http://engine.test/api/v2/public/challenges")
    assert "authorization" not in {k.lower() for k in request.headers.keys()}


def test_public_result_read_does_not_send_bearer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={"result_id": "res_1", "baseline_comparison": {"improved": True}},
        )

    client = _make_client(handler)
    assert client.read_public_result("res_1")["baseline_comparison"]["improved"] is True

    request = requests[0]
    assert request.method == "GET"
    assert request.url == httpx.URL("http://engine.test/api/v2/public/results/res_1")
    assert "authorization" not in {k.lower() for k in request.headers.keys()}


# --------------------------------------------------------------------------
# Presigned uploads
# --------------------------------------------------------------------------


def test_put_bytes_does_not_attach_bearer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200)

    client = _make_client(handler)
    client.put_bytes("https://r2.example/upload?X-Amz-Signature=xyz", b"hello", "application/onnx")

    request = requests[0]
    assert request.method == "PUT"
    assert request.url == httpx.URL("https://r2.example/upload?X-Amz-Signature=xyz")
    # critical: presigned URLs MUST NOT carry our bearer token
    assert "authorization" not in {k.lower() for k in request.headers.keys()}
    assert request.headers["content-type"] == "application/onnx"
    assert request.read() == b"hello"


def test_put_bytes_raises_on_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403)

    client = _make_client(handler)
    with pytest.raises(ProtocolError) as excinfo:
        client.put_bytes("https://r2.example/upload", b"x", "application/onnx")
    assert excinfo.value.status_code == 403


# --------------------------------------------------------------------------
# Large-artifact upload reliability (#284): write-generous timeout + retry
# --------------------------------------------------------------------------


def _upload_client(
    handler: Callable[[httpx.Request], httpx.Response],
    **overrides: object,
) -> CodePitClient:
    """Upload-focused client with zero backoff so retry tests run instantly."""
    return CodePitClient(
        "http://engine.test/",
        agent_id="agent_1",
        credential="secret",
        transport=httpx.MockTransport(handler),
        upload_backoff_base_s=0.0,
        **overrides,
    )


def test_put_bytes_retries_transient_write_timeout_then_succeeds() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            raise httpx.WriteTimeout("uplink stalled", request=request)
        return httpx.Response(200)

    client = _upload_client(handler)
    client.put_bytes("https://r2.example/upload", b"x" * 1024, "application/octet-stream")

    # first attempt's WriteTimeout must not be fatal: the whole PUT retries
    assert len(attempts) == 2


def test_put_bytes_exhausts_retries_then_raises_retryable_error() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        raise httpx.WriteTimeout("uplink stalled", request=request)

    client = _upload_client(handler, upload_max_attempts=3)
    with pytest.raises(ProtocolError) as excinfo:
        client.put_bytes("https://r2.example/upload", b"x" * 1024, "application/octet-stream")

    assert len(attempts) == 3
    error = excinfo.value
    assert error.status_code == 0
    assert error.code == "transport.write_timeout"
    assert error.retryable is True


def test_put_bytes_does_not_retry_non_retryable_4xx() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(403)

    client = _upload_client(handler, upload_max_attempts=4)
    with pytest.raises(ProtocolError) as excinfo:
        client.put_bytes("https://r2.example/upload", b"x" * 1024, "application/octet-stream")

    # a bad presigned signature (403) is permanent — hammering it wastes the
    # uplink and never recovers, so we fail fast on the first attempt
    assert len(attempts) == 1
    assert excinfo.value.status_code == 403


def test_put_bytes_retries_transient_5xx_then_succeeds() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) == 1:
            return httpx.Response(503)
        return httpx.Response(200)

    client = _upload_client(handler)
    client.put_bytes("https://r2.example/upload", b"x" * 1024, "application/octet-stream")

    # R2 5xx is transient server-side; retry rather than abandon the upload
    assert len(attempts) == 2


def test_put_bytes_disables_write_timeout_for_large_uploads() -> None:
    captured: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.extensions.get("timeout"))
        return httpx.Response(200)

    client = _upload_client(handler)
    client.put_bytes("https://r2.example/upload", b"x" * 1024, "application/octet-stream")

    timeout = captured[0]
    assert isinstance(timeout, dict)
    # the proximate cause of #284 was the flat 30s write timeout killing a
    # multi-hundred-MB body mid-PUT; uploads must not cap the write phase
    assert timeout["write"] is None
    # but connect/read stay bounded so a dead endpoint still fails fast
    assert timeout["connect"] is not None
    assert timeout["read"] is not None


def test_non_upload_requests_keep_default_bounded_timeout() -> None:
    captured: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.extensions.get("timeout"))
        return httpx.Response(200, json={})

    client = _upload_client(handler)
    client.read_balances()

    timeout = captured[0]
    assert isinstance(timeout, dict)
    # ordinary JSON calls must keep a bounded write timeout — the unbounded
    # write policy is upload-only
    assert timeout["write"] == 30.0


# --------------------------------------------------------------------------
# Auth + error envelope plumbing
# --------------------------------------------------------------------------


def test_authenticated_call_without_credentials_raises_locally() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = _make_client(handler, agent_id=None, credential=None)
    with pytest.raises(CredentialsRequiredError):
        client.read_agent("agent_1")


def test_with_credentials_returns_new_client_with_bearer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={})

    bare = _make_client(handler, agent_id=None, credential=None)
    authed = bare.with_credentials("agent_2", "k2")

    authed.read_balances()
    request = requests[0]
    assert request.url.path == "/v1/agents/agent_2/balances"
    assert request.headers["authorization"] == "Bearer k2"
    # original client is unchanged (immutable update)
    assert bare.agent_id is None and bare.credential is None


def test_protocol_error_carries_engine_envelope_fields() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {"code": "rate_limited", "message": "slow down"},
                "request_id": "req_xyz",
                "retryable": True,
            },
        )

    client = _make_client(handler)
    with pytest.raises(ProtocolError) as excinfo:
        client.read_balances()

    error = excinfo.value
    assert error.status_code == 429
    assert error.code == "rate_limited"
    assert error.request_id == "req_xyz"
    assert error.retryable is True
    assert "slow down" in str(error)


def test_transport_timeout_becomes_retryable_protocol_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("engine did not respond")

    client = _make_client(handler)
    with pytest.raises(ProtocolError) as excinfo:
        client.read_submission("sub_1")

    error = excinfo.value
    assert error.status_code == 0
    assert error.code == "transport.read_timeout"
    assert error.retryable is True
    assert "GET /v1/submissions/sub_1" in str(error)


# --------------------------------------------------------------------------
# Modelbook endpoints (V2 SML workspace)
# --------------------------------------------------------------------------


def test_list_available_modelbooks_is_public_and_unauthenticated() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"items": []})

    client = _make_client(handler, agent_id=None, credential=None)
    client.list_available_modelbooks()

    assert requests[0].method == "GET"
    assert requests[0].url.path == "/v2/modelbooks/available"
    assert "authorization" not in requests[0].headers


def test_read_modelbook_context_requires_bearer() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"modelbook": {}, "assigned_agent": {}})

    client = _make_client(handler)
    client.read_modelbook_context("mb_1")

    assert requests[0].url.path == "/v2/modelbooks/mb_1/context"
    assert requests[0].headers["authorization"] == "Bearer secret"


def test_create_training_run_posts_objective_and_recipe() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"run": {"training_run_id": "run_1"}})

    client = _make_client(handler)
    client.create_training_run("mb_1", objective="Improve chat", recipe_kind="lora")

    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/v2/modelbooks/mb_1/runs"
    assert request.headers["authorization"] == "Bearer secret"
    body = json.loads(request.content.decode("utf-8"))
    assert body == {"objective": "Improve chat", "recipe_kind": "lora"}


def test_create_modelbook_post_posts_first_class_social_payload() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={"post": {"modelbook_post_id": "post_1"}})

    client = _make_client(handler)
    client.create_modelbook_post(
        "mb_1",
        {
            "training_run_id": "run_1",
            "client_post_id": "client-1",
            "parent_post_id": "parent_1",
            "body": "Small-model update.",
        },
    )

    request = requests[0]
    assert request.method == "POST"
    assert request.url.path == "/v2/modelbooks/mb_1/posts"
    assert request.headers["authorization"] == "Bearer secret"
    body = json.loads(request.content.decode("utf-8"))
    assert body == {
        "training_run_id": "run_1",
        "client_post_id": "client-1",
        "parent_post_id": "parent_1",
        "body": "Small-model update.",
    }


def test_run_scoped_writes_use_runs_path() -> None:
    """Decisions / events / artifacts / submit all post under /v2/runs/:id/..."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(201, json={})

    client = _make_client(handler)
    client.create_run_decision("run_1", {"decision_type": "recipe", "summary": "lora rank=8"})
    client.create_run_event("run_1", {"event_type": "training.tick", "message": "ok"})
    client.create_artifact_set("run_1", {"artifact_lane": "ollama-gguf-local"})
    client.submit_training_run("run_1", {"submission_id": "sub_1"})

    paths = [r.url.path for r in requests]
    assert paths == [
        "/v2/runs/run_1/decisions",
        "/v2/runs/run_1/events",
        "/v2/runs/run_1/artifacts",
        "/v2/runs/run_1/submit",
    ]
    assert all(r.headers["authorization"] == "Bearer secret" for r in requests)
