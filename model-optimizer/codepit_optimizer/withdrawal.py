"""Owner-signed reward withdrawal for external agents (#270 Piece 4).

After an agent wins a sponsor competition and the reward settles into its
spendable balance (at challenge close), the **owner** withdraws it. The owner
signs the canonical withdrawal message (EIP-191 personal_sign) with the same
wallet bound as the agent's payout address at claim time, and posts it to
``POST /v2/agents/:id/withdrawals``. The engine recovers the signer and
authorizes the withdrawal only if recovery matches the bound payout address —
so the autonomous agent (runtime credential only) can never move funds.

This mirrors the claim flow in ``claim.py``; only the message domain and the
request endpoint differ.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Protocol

from .signer import AgentSigner

#: ±10 minutes, matching the engine's WITHDRAWAL_TIMESTAMP_SKEW_MS.
WITHDRAWAL_TIMESTAMP_SKEW_MS = 10 * 60 * 1000

_PROTOCOL_VERSION = "v1"


def build_withdrawal_message(
    *,
    agent_id: str,
    payout_address: str,
    amount_raw: str,
    client_withdrawal_id: str,
    timestamp_ms: int,
) -> str:
    """Byte-identical to the engine's ``buildWithdrawalMessage``.

    LF-joined, no trailing newline, payout address lowercased. The owner wallet
    signs *this exact string*; any drift breaks signature recovery on the engine.
    """
    return "\n".join(
        [
            "CodePit V2 Withdrawal Request",
            f"agent_id: {agent_id}",
            f"payout_address: {payout_address.lower()}",
            f"amount_raw: {amount_raw}",
            f"client_withdrawal_id: {client_withdrawal_id}",
            f"timestamp_ms: {timestamp_ms}",
        ]
    )


class _WithdrawalClient(Protocol):
    def request_withdrawal(self, agent_id: str, body: Mapping[str, Any]) -> Mapping[str, Any]: ...


def request_reward_withdrawal(
    client: _WithdrawalClient,
    *,
    agent_id: str,
    amount_raw: str,
    client_withdrawal_id: str,
    owner_private_key: str,
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Sign the withdrawal with the owner wallet and request payout.

    The payout destination is the owner wallet itself (the address recovered
    from ``owner_private_key``), which must equal the agent's bound payout
    address or the engine rejects the request.
    """
    owner = AgentSigner.from_private_key(owner_private_key)
    timestamp_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    message = build_withdrawal_message(
        agent_id=agent_id,
        payout_address=owner.address,
        amount_raw=amount_raw,
        client_withdrawal_id=client_withdrawal_id,
        timestamp_ms=timestamp_ms,
    )
    signature = owner.sign_message(message)
    response = client.request_withdrawal(
        agent_id,
        {
            "protocol_version": _PROTOCOL_VERSION,
            "owner_wallet_address": owner.address,
            "amount_raw": amount_raw,
            "client_withdrawal_id": client_withdrawal_id,
            "timestamp_ms": timestamp_ms,
            "signature": signature,
        },
    )
    return dict(response)
