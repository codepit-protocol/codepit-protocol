"""Persist and reload an agent's signer + runtime credential between runs.

The runtime credential is shown by the engine exactly once (at registration).
Losing it forces a credential rotation, which costs an extra round trip and
a fresh signer-bound challenge. So once an agent registers, we store its
signer private key and runtime credential under ``~/.codepit/agent.json``
(or an operator-chosen path) with file mode 0600.

Anyone with this file can act as the agent. Treat it like an SSH key.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_SESSION_PATH = Path.home() / ".codepit" / "agent.json"


@dataclass(frozen=True)
class AgentSession:
    base_url: str
    agent_id: str
    signer_private_key: str
    signer_address: str
    runtime_credential: str
    runtime_credential_id: str | None = None
    trust_tier: str | None = None
    agent_wallet_private_key: str | None = None
    agent_wallet_address: str | None = None


class SessionFileError(RuntimeError):
    """Raised when a session file is malformed or unreadable."""


def load_session(path: Path = DEFAULT_SESSION_PATH) -> AgentSession | None:
    """Return the persisted session, or ``None`` if the file does not exist."""
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SessionFileError(f"failed to read session at {path}: {error}") from error
    if not isinstance(raw, dict):
        raise SessionFileError(f"session at {path} is not a JSON object")
    return AgentSession(
        base_url=str(raw["base_url"]),
        agent_id=str(raw["agent_id"]),
        signer_private_key=str(raw["signer_private_key"]),
        signer_address=str(raw["signer_address"]),
        runtime_credential=str(raw["runtime_credential"]),
        runtime_credential_id=raw.get("runtime_credential_id"),
        trust_tier=raw.get("trust_tier"),
        agent_wallet_private_key=raw.get("agent_wallet_private_key"),
        agent_wallet_address=raw.get("agent_wallet_address"),
    )


def save_session(session: AgentSession, path: Path = DEFAULT_SESSION_PATH) -> None:
    """Write ``session`` atomically with mode 0600.

    Atomicity matters: a partial write on a crash leaves the agent unable
    to reconnect. We write to a temp file, fsync, and rename in place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(asdict(session), indent=2, sort_keys=True)
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)
    os.chmod(path, 0o600)
