"""Tests for owner-signed reward withdrawal (#270 Piece 4).

The owner withdraws a settled reward by signing the canonical withdrawal
message with the wallet bound as the agent's payout address, posted to
``POST /v1/agents/:id/withdrawals``.
"""

from __future__ import annotations

from typing import Any

import pytest

from codepit_optimizer.signer import AgentSigner, recover_signer_address
from codepit_optimizer.withdrawal import (
    build_withdrawal_message,
    request_reward_withdrawal,
)

_OWNER_KEY = "0x" + "22" * 32


def test_build_withdrawal_message_is_byte_identical_to_engine() -> None:
    message = build_withdrawal_message(
        agent_id="agent-7",
        payout_address="0xAbCDef0000000000000000000000000000000002",
        amount_raw="5000000000000000",
        client_withdrawal_id="wd-001",
        timestamp_ms=1700000000000,
    )
    assert message == (
        "CodePit V2 Withdrawal Request\n"
        "agent_id: agent-7\n"
        "payout_address: 0xabcdef0000000000000000000000000000000002\n"
        "amount_raw: 5000000000000000\n"
        "client_withdrawal_id: wd-001\n"
        "timestamp_ms: 1700000000000"
    )


class _CapturingWithdrawalClient:
    def __init__(self, response: dict[str, Any]) -> None:
        self._response = response
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def request_withdrawal(self, agent_id: str, body: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((agent_id, dict(body)))
        return self._response


def test_request_reward_withdrawal_sends_owner_signed_recoverable_request() -> None:
    owner = AgentSigner.from_private_key(_OWNER_KEY)
    client = _CapturingWithdrawalClient(
        {"withdrawal_request_id": "wr-1", "lifecycle_state": "PREPARED"}
    )

    result = request_reward_withdrawal(
        client,
        agent_id="agent-7",
        amount_raw="5000000000000000",
        client_withdrawal_id="wd-001",
        owner_private_key=_OWNER_KEY,
        now_ms=1700000000000,
    )

    assert result["lifecycle_state"] == "PREPARED"
    agent_id, body = client.calls[0]
    assert agent_id == "agent-7"
    assert body["protocol_version"] == "v1"
    assert body["owner_wallet_address"] == owner.address
    assert body["amount_raw"] == "5000000000000000"
    assert body["client_withdrawal_id"] == "wd-001"
    assert body["timestamp_ms"] == 1700000000000

    message = build_withdrawal_message(
        agent_id="agent-7",
        payout_address=owner.address,
        amount_raw="5000000000000000",
        client_withdrawal_id="wd-001",
        timestamp_ms=1700000000000,
    )
    assert recover_signer_address(message, body["signature"]).lower() == owner.address.lower()


def test_request_reward_withdrawal_defaults_timestamp_to_now(monkeypatch) -> None:
    client = _CapturingWithdrawalClient({"lifecycle_state": "PREPARED"})
    monkeypatch.setattr("codepit_optimizer.withdrawal.time.time", lambda: 1_700.0)

    request_reward_withdrawal(
        client,
        agent_id="a",
        amount_raw="1",
        client_withdrawal_id="wd",
        owner_private_key=_OWNER_KEY,
    )

    _, body = client.calls[0]
    assert body["timestamp_ms"] == 1_700_000
