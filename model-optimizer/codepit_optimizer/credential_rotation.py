from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

from .orchestrator import OrchestratorError
from .protocol import CodePitClient
from .session import DEFAULT_SESSION_PATH, AgentSession, load_session, save_session
from .signer import AgentSigner


@dataclass(frozen=True)
class CredentialRotationConfig:
    base_url: str
    agent_id: str | None = None
    private_key: str | None = None
    session_path: Path | None = DEFAULT_SESSION_PATH
    trust_tier: str | None = None


@dataclass(frozen=True)
class CredentialRotationResult:
    agent_id: str
    signer_address: str
    credential_id: str
    runtime_credential: str
    superseded_credential_id: str | None
    session_path: str | None


def hash_rotation_intent(agent_id: str, signer_address: str) -> str:
    canonical = f"rotate:{agent_id}:{signer_address.lower()}"
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def rotate_optimizer_credentials(
    config: CredentialRotationConfig,
) -> CredentialRotationResult:
    signer, agent_id, persisted = _resolve_rotation_context(config)
    client = CodePitClient(config.base_url)
    intent_hash = hash_rotation_intent(agent_id, signer.address)
    challenge = client.request_auth_challenge(
        {
            "protocol_version": "v1",
            "agent_signer_address": signer.address,
            "registration_payload_hash": intent_hash,
        },
    )
    signature = signer.sign_message(challenge["message"])
    rotated = client.rotate_credentials(
        agent_id,
        {
            "protocol_version": "v1",
            "challenge_id": challenge["challenge_id"],
            "nonce": challenge["nonce"],
            "timestamp_ms": int(time.time() * 1000),
            "signature": signature,
            "agent_signer_address": signer.address,
        },
    )
    credential = rotated["credential"]
    session_path = str(config.session_path) if config.session_path is not None else None
    if config.session_path is not None:
        save_session(
            AgentSession(
                base_url=config.base_url,
                agent_id=agent_id,
                signer_private_key=signer.private_key,
                signer_address=signer.address,
                runtime_credential=credential["secret"],
                runtime_credential_id=credential.get("id"),
                trust_tier=config.trust_tier
                or (persisted.trust_tier if persisted is not None else None),
                agent_wallet_private_key=(
                    persisted.agent_wallet_private_key if persisted is not None else None
                ),
                agent_wallet_address=(
                    persisted.agent_wallet_address if persisted is not None else None
                ),
            ),
            path=config.session_path,
        )

    return CredentialRotationResult(
        agent_id=agent_id,
        signer_address=signer.address,
        credential_id=credential["id"],
        runtime_credential=credential["secret"],
        superseded_credential_id=rotated.get("superseded_credential_id"),
        session_path=session_path,
    )


def _resolve_rotation_context(
    config: CredentialRotationConfig,
) -> tuple[AgentSigner, str, AgentSession | None]:
    persisted = None
    if config.session_path is not None:
        persisted = load_session(config.session_path)
        if persisted is not None and persisted.base_url != config.base_url:
            persisted = None

    if config.private_key:
        signer = AgentSigner.from_private_key(config.private_key)
    elif persisted is not None:
        signer = AgentSigner.from_private_key(persisted.signer_private_key)
    else:
        raise OrchestratorError(
            "credential rotation requires --private-key or a persisted session "
            "matching --base-url",
        )

    agent_id = config.agent_id or (persisted.agent_id if persisted is not None else None)
    if not agent_id:
        raise OrchestratorError(
            "credential rotation requires --agent-id or a persisted session "
            "matching --base-url",
        )

    return signer, agent_id, persisted
