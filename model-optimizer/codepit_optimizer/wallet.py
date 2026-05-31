"""Agent wallet helpers for CodePit V2 external agents.

The agent wallet is separate from the protocol signer. The signer authorizes
registration and credential rotation; the agent wallet is the account the
agent will use in the A2A economy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .signer import AgentSigner, recover_signer_address


@dataclass(frozen=True)
class AgentWallet:
    private_key: str
    address: str

    @classmethod
    def from_private_key(cls, private_key: str) -> "AgentWallet":
        signer = AgentSigner.from_private_key(private_key)
        return cls(private_key=signer.private_key, address=signer.address)

    @classmethod
    def ephemeral(cls) -> "AgentWallet":
        signer = AgentSigner.ephemeral()
        return cls(private_key=signer.private_key, address=signer.address)

    def sign_message(self, message: str) -> str:
        return AgentSigner.from_private_key(self.private_key).sign_message(message)


def build_agent_wallet_binding_message(
    *,
    agent_signer_address: str,
    agent_wallet_address: str,
    registration_payload_hash: str,
    timestamp_ms: int,
) -> str:
    return "\n".join(
        [
            "CodePit V2 Agent Wallet Binding",
            f"agent_signer: {agent_signer_address.lower()}",
            f"agent_wallet: {agent_wallet_address.lower()}",
            f"registration_payload_hash: {registration_payload_hash}",
            f"timestamp_ms: {timestamp_ms}",
        ]
    )


def recover_agent_wallet_address(message: str, signature_hex: str) -> str:
    return recover_signer_address(message, signature_hex)
