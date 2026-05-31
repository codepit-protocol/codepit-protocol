"""Tests for owner-claim payout binding (slice F, #272).

The owner binds a payout wallet by signing the canonical claim message with
the wallet they control and posting it (with the single-use claim_token from
registration) to the existing ``POST /v1/agents/:id/claim`` endpoint. No new
engine endpoint — this is the kit side of the engine's owner-claim flow.
"""

from __future__ import annotations

from typing import Any

import pytest

from codepit_optimizer.claim import (
    PayoutWalletError,
    build_claim_message,
    claim_agent_payout,
)
from codepit_optimizer.signer import AgentSigner, recover_signer_address


def test_build_claim_message_is_byte_identical_to_engine() -> None:
    message = build_claim_message(
        agent_id="agent-123",
        agent_signer_address="0xAbCDef0000000000000000000000000000000001",
        claim_token="tok_plain",
        timestamp_ms=1700000000000,
    )

    # must match engine buildClaimMessage exactly: LF-joined, no trailing
    # newline, signer lowercased (claims/claim-message.ts)
    assert message == (
        "CodePit V2 Claim\n"
        "agent_id: agent-123\n"
        "agent_signer: 0xabcdef0000000000000000000000000000000001\n"
        "claim_token: tok_plain\n"
        "timestamp_ms: 1700000000000"
    )


class _CapturingClaimClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def claim_agent(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((agent_id, dict(body)))
        return self._response


_OWNER_KEY = "0x" + "11" * 32


def test_claim_agent_payout_sends_owner_signed_recoverable_request() -> None:
    owner = AgentSigner.from_private_key(_OWNER_KEY)
    client = _CapturingClaimClient(
        {"agent_id": "agent-9", "payout_address": owner.address.lower(), "claim_status": "claimed"}
    )

    agent_signer_address = "0xAbCd" + "0" * 36

    result = claim_agent_payout(
        client,
        agent_id="agent-9",
        agent_signer_address=agent_signer_address,
        claim_token="tok_xyz",
        owner_private_key=_OWNER_KEY,
        now_ms=1700000000000,
    )

    assert result["claim_status"] == "claimed"
    agent_id, body = client.calls[0]
    assert agent_id == "agent-9"
    assert body["protocol_version"] == "v1"
    assert body["claim_token"] == "tok_xyz"
    # the payout address bound is the owner wallet itself (the recovered signer)
    assert body["owner_wallet_address"] == owner.address
    assert body["timestamp_ms"] == 1700000000000

    # the signature must recover to the owner wallet against the exact message
    message = build_claim_message(
        agent_id="agent-9",
        agent_signer_address=agent_signer_address,
        claim_token="tok_xyz",
        timestamp_ms=1700000000000,
    )
    assert recover_signer_address(message, body["signature"]).lower() == owner.address.lower()


def test_claim_refuses_binding_the_agents_own_throwaway_wallet() -> None:
    # Footgun guard: binding the autonomous agent's own ephemeral signer/wallet
    # as the payout wallet defeats the security model (the agent could then move
    # funds) AND risks loss (the ephemeral key is rarely backed up). Refuse it,
    # and do NOT send the claim request.
    owner = AgentSigner.from_private_key(_OWNER_KEY)
    client = _CapturingClaimClient({"claim_status": "claimed"})

    with pytest.raises(PayoutWalletError) as excinfo:
        claim_agent_payout(
            client,
            agent_id="agent-9",
            agent_signer_address="0x" + "a" * 40,
            claim_token="tok_xyz",
            owner_private_key=_OWNER_KEY,
            # the agent's own addresses — owner here equals one of them
            forbidden_payout_addresses=[owner.address],
            now_ms=1700000000000,
        )

    assert "control" in str(excinfo.value).lower()
    assert client.calls == []  # nothing posted to the engine


def test_claim_forbidden_payout_match_is_case_insensitive() -> None:
    owner = AgentSigner.from_private_key(_OWNER_KEY)
    client = _CapturingClaimClient({"claim_status": "claimed"})

    with pytest.raises(PayoutWalletError):
        claim_agent_payout(
            client,
            agent_id="agent-9",
            agent_signer_address="0x" + "a" * 40,
            claim_token="tok_xyz",
            owner_private_key=_OWNER_KEY,
            forbidden_payout_addresses=[owner.address.upper()],
        )
    assert client.calls == []


def test_claim_allows_a_distinct_owner_wallet() -> None:
    owner = AgentSigner.from_private_key(_OWNER_KEY)
    other = AgentSigner.from_private_key("0x" + "22" * 32)
    client = _CapturingClaimClient(
        {"agent_id": "agent-9", "payout_address": owner.address.lower(), "claim_status": "claimed"}
    )

    result = claim_agent_payout(
        client,
        agent_id="agent-9",
        agent_signer_address="0x" + "a" * 40,
        claim_token="tok_xyz",
        owner_private_key=_OWNER_KEY,
        forbidden_payout_addresses=[other.address],  # owner != agent's own wallets
        now_ms=1700000000000,
    )
    assert result["claim_status"] == "claimed"
    assert len(client.calls) == 1


def test_claim_agent_payout_defaults_timestamp_to_now(monkeypatch) -> None:
    client = _CapturingClaimClient({"claim_status": "claimed"})
    monkeypatch.setattr("codepit_optimizer.claim.time.time", lambda: 1_700.0)

    claim_agent_payout(
        client,
        agent_id="a",
        agent_signer_address="0x" + "a" * 40,
        claim_token="t",
        owner_private_key=_OWNER_KEY,
    )

    _, body = client.calls[0]
    assert body["timestamp_ms"] == 1_700_000
