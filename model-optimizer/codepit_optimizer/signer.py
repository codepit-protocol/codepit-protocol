"""Agent signer for V2 registration and credential rotation.

Produces signatures byte-compatible with viem's
``signer.signMessage({ message })`` (EIP-191 ``personal_sign``), which is
what the engine recovers from in
``src/v2/protocol/registration/service.ts``.

We use ``eth-keys`` + ``eth-utils`` directly rather than ``eth-account``
because eth-account 0.13+ pulls in ``ckzg`` (EIP-4844 blob transactions),
which has fragile native wheels on Apple Silicon under Rosetta. We never
need 4844 blobs — only personal_sign — so the lower-level libs are a
better fit.

Two modes:
- ``AgentSigner.from_private_key("0x...")``: reuse an existing signer
  (typical for second runs or operator-controlled signers).
- ``AgentSigner.ephemeral()``: generate a fresh keypair per process. Used
  when the agent has no persisted signer yet.
"""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from eth_keys import keys
from eth_utils import keccak

PERSONAL_SIGN_PREFIX = b"\x19Ethereum Signed Message:\n"


@dataclass(frozen=True)
class AgentSigner:
    private_key: str  # 0x-prefixed lowercase hex (66 chars total)
    address: str  # checksum 0x-prefixed address

    @classmethod
    def from_private_key(cls, private_key: str) -> "AgentSigner":
        normalized = _normalize_private_key(private_key)
        priv = keys.PrivateKey(bytes.fromhex(normalized[2:]))
        return cls(private_key=normalized, address=priv.public_key.to_checksum_address())

    @classmethod
    def ephemeral(cls) -> "AgentSigner":
        return cls.from_private_key("0x" + secrets.token_hex(32))

    def sign_message(self, message: str) -> str:
        """Return a 0x-prefixed personal_sign signature with v in {27, 28}.

        viem.verifyMessage on the engine side accepts both v ∈ {0,1} and
        v ∈ {27,28}; we emit the conventional 27/28 form so the bytes are
        identical to what TS reference agents send.
        """
        message_hash = _eip191_message_hash(message)
        priv = keys.PrivateKey(bytes.fromhex(self.private_key[2:]))
        signature = priv.sign_msg_hash(message_hash)
        return _serialize_signature(signature)


def recover_signer_address(message: str, signature_hex: str) -> str:
    """Recover the checksum address that produced ``signature_hex`` for ``message``.

    Used in tests to verify round-trip and by the orchestrator to sanity-check
    a signature before sending it to the engine.
    """
    signature_bytes = _signature_bytes(signature_hex)
    canonical_v = _canonical_v(signature_bytes[64])
    canonical_signature = signature_bytes[:64] + bytes([canonical_v])
    parsed = keys.Signature(canonical_signature)
    public_key = parsed.recover_public_key_from_msg_hash(_eip191_message_hash(message))
    return public_key.to_checksum_address()


def _eip191_message_hash(message: str) -> bytes:
    body = message.encode("utf-8")
    return keccak(PERSONAL_SIGN_PREFIX + str(len(body)).encode("ascii") + body)


def _serialize_signature(signature: keys.Signature) -> str:
    raw = signature.to_bytes()
    r = raw[:32]
    s = raw[32:64]
    v = raw[64]
    canonical_v = v if v >= 27 else v + 27
    return "0x" + r.hex() + s.hex() + format(canonical_v, "02x")


def _signature_bytes(signature_hex: str) -> bytes:
    body = signature_hex[2:] if signature_hex.lower().startswith("0x") else signature_hex
    if len(body) != 130:
        raise ValueError("signature must be 65 bytes (130 hex characters)")
    return bytes.fromhex(body)


def _canonical_v(v: int) -> int:
    if v in (0, 1):
        return v
    if v in (27, 28):
        return v - 27
    raise ValueError(f"unsupported v byte: {v}")


def _normalize_private_key(value: str) -> str:
    trimmed = value.strip()
    if not trimmed:
        raise ValueError("private key is empty")
    body = trimmed[2:] if trimmed.lower().startswith("0x") else trimmed
    if len(body) != 64:
        raise ValueError("private key must be 32 bytes (64 hex characters)")
    try:
        int(body, 16)
    except ValueError as error:
        raise ValueError("private key must be hex") from error
    return f"0x{body.lower()}"
