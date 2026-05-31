from __future__ import annotations

import hashlib
import json
from typing import Callable
from unittest.mock import patch

import httpx
import pytest

from codepit_optimizer.credential_rotation import (
    CredentialRotationConfig,
    hash_rotation_intent,
    rotate_optimizer_credentials,
)
from codepit_optimizer.orchestrator import OrchestratorError
from codepit_optimizer.protocol import CodePitClient
from codepit_optimizer.session import AgentSession, load_session, save_session
from codepit_optimizer.signer import AgentSigner, recover_signer_address


class FakeRotationEngine:
    def __init__(self, signer: AgentSigner, agent_id: str = "agent_pyopt_1") -> None:
        self.signer = signer
        self.agent_id = agent_id
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.read() or b"{}")

        if request.method == "POST" and request.url.path == "/v1/agents/auth/challenge":
            assert "authorization" not in {key.lower() for key in request.headers.keys()}
            assert body["registration_payload_hash"] == hash_rotation_intent(
                self.agent_id,
                self.signer.address,
            )
            return httpx.Response(
                201,
                json={
                    "challenge_id": "ch_rotate_1",
                    "nonce": "nonce_rotate_1",
                    "message": "rotate challenge",
                    "expires_at": "2026-05-06T00:05:00Z",
                },
            )

        rotate_path = f"/v1/agents/{self.agent_id}/credentials/rotate"
        if request.method == "POST" and request.url.path == rotate_path:
            assert "authorization" not in {key.lower() for key in request.headers.keys()}
            recovered = recover_signer_address("rotate challenge", body["signature"])
            assert recovered == self.signer.address
            assert body["challenge_id"] == "ch_rotate_1"
            assert body["nonce"] == "nonce_rotate_1"
            return httpx.Response(
                201,
                json={
                    "credential": {"id": "cred_new", "secret": "new_secret"},
                    "superseded_credential_id": "cred_old",
                },
            )

        return httpx.Response(404, json={"error": {"code": "not_found"}})


def _stub_client(engine: FakeRotationEngine) -> Callable[..., CodePitClient]:
    transport = httpx.MockTransport(engine.handler)

    def factory(base_url: str, agent_id=None, credential=None) -> CodePitClient:
        return CodePitClient(
            base_url,
            agent_id=agent_id,
            credential=credential,
            transport=transport,
        )

    return factory


def test_hash_rotation_intent_matches_engine_shape_and_lowercases_signer() -> None:
    signer = "0xAbCdEf0000000000000000000000000000000001"
    canonical = "rotate:agent_1:0xabcdef0000000000000000000000000000000001"
    expected = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    assert hash_rotation_intent("agent_1", signer) == expected
    assert hash_rotation_intent("agent_1", signer.lower()) == expected


def test_rotate_optimizer_credentials_uses_persisted_session_and_saves_new_secret(
    tmp_path,
) -> None:
    signer = AgentSigner.from_private_key("0x" + "11" * 32)
    session_path = tmp_path / "agent.json"
    save_session(
        AgentSession(
            base_url="http://engine.fake",
            agent_id="agent_pyopt_1",
            signer_private_key=signer.private_key,
            signer_address=signer.address,
            runtime_credential="old_secret",
            runtime_credential_id="cred_old",
            trust_tier="Sandbox",
            agent_wallet_private_key="0x" + "22" * 32,
            agent_wallet_address="0x" + "b" * 40,
        ),
        path=session_path,
    )
    engine = FakeRotationEngine(signer)

    with patch(
        "codepit_optimizer.credential_rotation.CodePitClient",
        side_effect=_stub_client(engine),
    ):
        result = rotate_optimizer_credentials(
            CredentialRotationConfig(
                base_url="http://engine.fake",
                session_path=session_path,
            )
        )

    assert result.agent_id == "agent_pyopt_1"
    assert result.credential_id == "cred_new"
    assert result.runtime_credential == "new_secret"
    assert result.superseded_credential_id == "cred_old"
    saved = load_session(session_path)
    assert saved is not None
    assert saved.runtime_credential == "new_secret"
    assert saved.runtime_credential_id == "cred_new"
    assert saved.agent_wallet_private_key == "0x" + "22" * 32
    assert saved.agent_wallet_address == "0x" + "b" * 40
    assert [request.url.path for request in engine.requests] == [
        "/v1/agents/auth/challenge",
        "/v1/agents/agent_pyopt_1/credentials/rotate",
    ]


def test_rotate_optimizer_credentials_requires_local_context_before_network() -> None:
    with pytest.raises(OrchestratorError, match="private-key or a persisted session"):
        rotate_optimizer_credentials(
            CredentialRotationConfig(
                base_url="http://engine.fake",
                session_path=None,
            )
        )
