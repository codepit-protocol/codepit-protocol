"""Verify Python canonicalization + hash matches engine TS fixtures byte-for-byte.

Fixtures are produced by ``engine/scripts/emit-payload-hash-fixtures.ts``,
which calls the production canonicalizer in
``engine/src/v2/protocol/registration/payload-hash.ts``. If these tests
ever fail it means the runtime hash and the server-recomputed hash will
disagree and registration will fail with ``auth.invalid_signature``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codepit_optimizer.payload_hash import (
    PayloadCanonicalizationError,
    canonicalize_registration_payload,
    hash_registration_payload,
)

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "registration_payload_hash.json"


def _load_fixtures() -> list[dict]:
    return json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("fixture", _load_fixtures(), ids=lambda f: f["name"])
def test_canonicalization_matches_engine(fixture: dict) -> None:
    canonical = canonicalize_registration_payload(fixture["payload"])
    assert canonical == fixture["canonical"]


@pytest.mark.parametrize("fixture", _load_fixtures(), ids=lambda f: f["name"])
def test_hash_matches_engine(fixture: dict) -> None:
    assert hash_registration_payload(fixture["payload"]) == fixture["hash"]


def test_lowercases_mixed_case_signer() -> None:
    mixed = {
        "protocol_version": "v1",
        "agent_signer_address": "0xAbCdEf0000000000000000000000000000000001",
        "agent": {"display_name": "x", "mode": "external"},
        "capabilities": {
            "declared_artifact_lanes": ["onnx-browser-webgpu"],
            "declared_at_version": "v1",
            "declared_model_classes": ["encoder-text-small"],
            "declared_runtimes": ["onnxruntime-web-webgpu"],
            "optimization_methods": ["graph-optimization"],
        },
    }
    lowered = {**mixed, "agent_signer_address": mixed["agent_signer_address"].lower()}
    assert hash_registration_payload(mixed) == hash_registration_payload(lowered)


def test_agent_wallet_fields_are_bound_into_engine_hash() -> None:
    base = {
        "protocol_version": "v1",
        "agent_signer_address": "0x" + "a" * 40,
        "agent": {"display_name": "x", "mode": "external"},
        "capabilities": {
            "declared_artifact_lanes": ["onnx-browser-webgpu"],
            "declared_at_version": "v1",
            "declared_model_classes": ["encoder-text-small"],
            "declared_runtimes": ["onnxruntime-web-webgpu"],
            "optimization_methods": ["graph-optimization"],
        },
    }
    with_wallet = {
        **base,
        "agent_wallet": {
            "address": "0xBBBBbbbbBBBBbbbbBBBBbbbbBBBBbbbbBBBBbbbb",
            "chain_id": 84532,
            "network": "base-sepolia",
            "wallet_provider": "local",
            "custody_mode": "agent_local",
        },
    }

    canonical = canonicalize_registration_payload(with_wallet)

    assert '"agent_wallet"' in canonical
    assert "0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb" in canonical
    assert canonical != canonicalize_registration_payload(base)
    assert hash_registration_payload(with_wallet) != hash_registration_payload(base)


def test_rejects_missing_top_level_key() -> None:
    with pytest.raises(PayloadCanonicalizationError):
        canonicalize_registration_payload(
            {
                "protocol_version": "v1",
                "agent_signer_address": "0x" + "a" * 40,
                "agent": {"display_name": "x", "mode": "external"},
                # capabilities omitted
            },
        )


def test_rejects_unknown_agent_mode() -> None:
    with pytest.raises(PayloadCanonicalizationError):
        canonicalize_registration_payload(
            {
                "protocol_version": "v1",
                "agent_signer_address": "0x" + "a" * 40,
                "agent": {"display_name": "x", "mode": "rogue"},
                "capabilities": {
                    "declared_artifact_lanes": ["onnx-browser-webgpu"],
                    "declared_at_version": "v1",
                    "declared_model_classes": ["encoder-text-small"],
                    "declared_runtimes": ["onnxruntime-web-webgpu"],
                    "optimization_methods": ["graph-optimization"],
                },
            },
        )
