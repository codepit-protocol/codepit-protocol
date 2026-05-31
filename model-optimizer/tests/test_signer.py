"""Round-trip tests for the agent signer.

Recoverability of the signer address is the only invariant we need: the
engine recovers the signer with viem.verifyMessage, which uses the same
EIP-191 personal_sign hash, so as long as our local recovery returns the
issued address the engine will accept the signature.
"""

from __future__ import annotations

import re

import pytest

from codepit_optimizer.signer import AgentSigner, recover_signer_address


def test_ephemeral_round_trip() -> None:
    signer = AgentSigner.ephemeral()
    message = "challenge message"
    signature = signer.sign_message(message)

    assert re.fullmatch(r"0x[0-9a-fA-F]{130}", signature)
    assert recover_signer_address(message, signature) == signer.address


def test_from_private_key_normalizes_hex() -> None:
    raw = "deadbeef" * 8  # 64 hex chars, no 0x
    signer = AgentSigner.from_private_key(raw)

    assert signer.private_key == f"0x{raw}"
    assert signer.address.startswith("0x")
    again = AgentSigner.from_private_key(f"0x{raw}")
    assert signer.address == again.address


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "0x",
        "0xnothex" + "0" * 56,
        "0x" + "a" * 63,
        "0x" + "a" * 65,
    ],
)
def test_rejects_invalid_private_keys(bad: str) -> None:
    with pytest.raises(ValueError):
        AgentSigner.from_private_key(bad)


def test_signature_round_trip_with_known_key() -> None:
    private_key = "0x" + "11" * 32
    signer = AgentSigner.from_private_key(private_key)
    signature = signer.sign_message("hello")
    assert recover_signer_address("hello", signature) == signer.address


def test_v_byte_is_27_or_28() -> None:
    """viem emits v ∈ {27, 28}; we should match so signatures are byte-equal."""
    signer = AgentSigner.from_private_key("0x" + "22" * 32)
    signature = signer.sign_message("v-byte check")
    v_byte = int(signature[-2:], 16)
    assert v_byte in (27, 28)
