"""Canonicalization + SHA-256 hashing of the V2 registration payload.

Mirrors `engine/src/v2/protocol/registration/payload-hash.ts` byte-for-byte so
the Python optimizer agent can produce a `registration_payload_hash` that the
engine recreates and verifies. The two paths MUST agree on every byte.

Rules encoded:
- signer and agent-wallet addresses are lowercased before hashing
- object keys are emitted in sorted order at every depth
- top-level layout is fixed (agent, agent_signer_address, optional
  agent_wallet, capabilities, protocol_version) and inner capability keys are
  emitted only when present
- array order is preserved (declarations are positional)
- ``declared_limits`` is included whenever the key is present, even if its
  value is ``None`` (matches the TS `!== undefined` check)
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

ProtocolVersion = str

_AGENT_MODES = ("external", "managed")


class PayloadCanonicalizationError(ValueError):
    """Raised when a registration payload cannot be canonicalized."""


def canonicalize_registration_payload(payload: Mapping[str, Any]) -> str:
    """Return the canonical JSON string the engine will hash.

    The shape is fixed so we reject unexpected inputs early — bytes that the
    engine never sees here would silently desync the on-wire hash from the
    server-recomputed hash and break registration.
    """
    _require_keys(payload, ("protocol_version", "agent_signer_address", "agent", "capabilities"))

    agent = payload["agent"]
    _require_keys(agent, ("display_name", "mode"))
    if agent["mode"] not in _AGENT_MODES:
        raise PayloadCanonicalizationError(
            f"agent.mode must be one of {_AGENT_MODES}, got {agent['mode']!r}",
        )

    canonical = {
        "agent": {
            "display_name": agent["display_name"],
            "mode": agent["mode"],
        },
        "agent_signer_address": str(payload["agent_signer_address"]).lower(),
        "capabilities": _canonicalize_capabilities(payload["capabilities"]),
        "protocol_version": payload["protocol_version"],
    }
    if "agent_wallet" in payload and payload["agent_wallet"] is not None:
        canonical["agent_wallet"] = _canonicalize_agent_wallet(payload["agent_wallet"])
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def hash_registration_payload(payload: Mapping[str, Any]) -> str:
    """Return ``"sha256:<hex>"`` matching the engine's hashRegistrationPayload."""
    canonical = canonicalize_registration_payload(payload)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _canonicalize_capabilities(caps: Mapping[str, Any]) -> dict[str, Any]:
    _require_keys(
        caps,
        (
            "declared_artifact_lanes",
            "declared_at_version",
            "declared_model_classes",
            "declared_runtimes",
            "optimization_methods",
        ),
    )
    out: dict[str, Any] = {
        "declared_artifact_lanes": list(caps["declared_artifact_lanes"]),
        "declared_at_version": caps["declared_at_version"],
        "declared_model_classes": list(caps["declared_model_classes"]),
        "declared_runtimes": list(caps["declared_runtimes"]),
        "optimization_methods": list(caps["optimization_methods"]),
    }
    if "declared_limits" in caps:
        out["declared_limits"] = caps["declared_limits"]
    return out


def _canonicalize_agent_wallet(wallet: Mapping[str, Any]) -> dict[str, Any]:
    _require_keys(
        wallet,
        (
            "address",
            "chain_id",
            "custody_mode",
            "network",
            "wallet_provider",
        ),
    )
    out: dict[str, Any] = {
        "address": str(wallet["address"]).lower(),
        "chain_id": wallet["chain_id"],
        "custody_mode": wallet["custody_mode"],
        "network": wallet["network"],
        "wallet_provider": wallet["wallet_provider"],
    }
    if "provider_wallet_id" in wallet:
        out["provider_wallet_id"] = wallet["provider_wallet_id"]
    if "policy_ref" in wallet:
        out["policy_ref"] = wallet["policy_ref"]
    return out


def _require_keys(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> None:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise PayloadCanonicalizationError(
            f"missing required keys: {sorted(missing)}",
        )
