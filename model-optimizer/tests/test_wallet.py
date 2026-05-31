from codepit_optimizer.wallet import (
    AgentWallet,
    build_agent_wallet_binding_message,
    recover_agent_wallet_address,
)


def test_local_wallet_generation_returns_evm_address() -> None:
    wallet = AgentWallet.ephemeral()
    assert wallet.address.startswith("0x")
    assert len(wallet.address) == 42
    assert wallet.private_key.startswith("0x")


def test_import_from_private_key_is_deterministic() -> None:
    private_key = "0x" + "11" * 32
    assert AgentWallet.from_private_key(private_key) == AgentWallet.from_private_key(private_key)


def test_wallet_binding_signature_recovers_to_wallet_address() -> None:
    wallet = AgentWallet.from_private_key("0x" + "22" * 32)
    message = build_agent_wallet_binding_message(
        agent_signer_address="0x" + "a" * 40,
        agent_wallet_address=wallet.address,
        registration_payload_hash="sha256:abc",
        timestamp_ms=123,
    )
    signature = wallet.sign_message(message)
    assert recover_agent_wallet_address(message, signature) == wallet.address


def test_wallet_binding_message_matches_engine_shape() -> None:
    message = build_agent_wallet_binding_message(
        agent_signer_address="0xABCDEFabcdefABCDEFabcdefABCDEFabcdefABCD",
        agent_wallet_address="0x111111111111111111111111111111111111AAAA",
        registration_payload_hash="sha256:abc",
        timestamp_ms=123,
    )
    assert message == "\n".join(
        [
            "CodePit V2 Agent Wallet Binding",
            "agent_signer: 0xabcdefabcdefabcdefabcdefabcdefabcdefabcd",
            "agent_wallet: 0x111111111111111111111111111111111111aaaa",
            "registration_payload_hash: sha256:abc",
            "timestamp_ms: 123",
        ]
    )
