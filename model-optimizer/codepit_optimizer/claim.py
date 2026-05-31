"""Owner-claim payout binding for external agents (slice F, #272).

Binds an agent's ``payout_address`` so a verified sponsor submission can
actually earn — today an unclaimed agent silently forfeits its reward
(settlement marks ``missing_payout_address``). This is the kit side of the
engine's existing owner-claim flow (``POST /v1/agents/:id/claim``); it adds
**no new engine endpoint**.

The human owner signs the canonical claim message (EIP-191 personal_sign)
with the wallet they control. That signed wallet address becomes the agent's
payout address *and* the only address authorized to withdraw later — so the
autonomous agent (which holds only a runtime credential) can never move funds.
"""

from __future__ import annotations

import time
from typing import Any, Iterable, Mapping, Protocol

from .signer import AgentSigner

#: ±10 minutes, matching the engine's CLAIM_TIMESTAMP_SKEW_MS.
CLAIM_TIMESTAMP_SKEW_MS = 10 * 60 * 1000

_PROTOCOL_VERSION = "v1"


class PayoutWalletError(RuntimeError):
    """Raised when the proposed payout wallet is unsafe to bind.

    Fires when the owner key resolves to one of the agent's own auto-generated
    addresses (signer / agent wallet). Binding a throwaway key as the payout
    wallet defeats the security model (the autonomous agent could then move
    funds) and risks permanently locking rewards, since those ephemeral keys
    are rarely backed up. The owner must bind a wallet they actually control
    and have backed up.
    """


def build_claim_message(
    *,
    agent_id: str,
    agent_signer_address: str,
    claim_token: str,
    timestamp_ms: int,
) -> str:
    """Byte-identical to the engine's ``buildClaimMessage``.

    LF-joined, no trailing newline, signer address lowercased. The owner wallet
    signs *this exact string*; any drift breaks signature recovery on the engine.
    """
    return "\n".join(
        [
            "CodePit V2 Claim",
            f"agent_id: {agent_id}",
            f"agent_signer: {agent_signer_address.lower()}",
            f"claim_token: {claim_token}",
            f"timestamp_ms: {timestamp_ms}",
        ]
    )


class _ClaimClient(Protocol):
    def claim_agent(self, agent_id: str, body: Mapping[str, Any]) -> Mapping[str, Any]: ...


def claim_agent_payout(
    client: _ClaimClient,
    *,
    agent_id: str,
    agent_signer_address: str,
    claim_token: str,
    owner_private_key: str,
    forbidden_payout_addresses: Iterable[str] = (),
    now_ms: int | None = None,
) -> dict[str, Any]:
    """Sign the claim with the owner wallet and bind the payout address.

    The payout address bound is the owner wallet's own address (the recovered
    signer), so the caller does not pass it separately — it is derived from
    ``owner_private_key`` and sent as ``owner_wallet_address``.

    ``forbidden_payout_addresses`` is a safety guard: pass the agent's own
    auto-generated addresses (signer + agent wallet). If the owner key resolves
    to one of them, the claim is refused (``PayoutWalletError``) before any
    request is sent — binding a throwaway key as payout would let the agent
    move funds and risks locking rewards in an unrecoverable wallet.
    """
    owner = AgentSigner.from_private_key(owner_private_key)
    forbidden = {addr.lower() for addr in forbidden_payout_addresses if addr}
    if owner.address.lower() in forbidden:
        raise PayoutWalletError(
            "refusing to bind the payout wallet: the owner key resolves to one "
            f"of the agent's own auto-generated wallets ({owner.address}). Bind a "
            "wallet you personally control and have backed up — rewards are paid "
            "to this address and only its private key can move them."
        )
    timestamp_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    message = build_claim_message(
        agent_id=agent_id,
        agent_signer_address=agent_signer_address,
        claim_token=claim_token,
        timestamp_ms=timestamp_ms,
    )
    signature = owner.sign_message(message)
    response = client.claim_agent(
        agent_id,
        {
            "protocol_version": _PROTOCOL_VERSION,
            "claim_token": claim_token,
            "owner_wallet_address": owner.address,
            "timestamp_ms": timestamp_ms,
            "signature": signature,
        },
    )
    return dict(response)
