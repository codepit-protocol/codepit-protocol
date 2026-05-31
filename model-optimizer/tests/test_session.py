"""Session persistence tests."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from codepit_optimizer.session import (
    AgentSession,
    SessionFileError,
    load_session,
    save_session,
)


def _make_session() -> AgentSession:
    return AgentSession(
        base_url="https://engine.codepit.fun",
        agent_id="agent_1",
        signer_private_key="0x" + "11" * 32,
        signer_address="0xabc",
        runtime_credential="rt_secret",
        runtime_credential_id="cred_1",
        trust_tier="Sandbox",
        agent_wallet_private_key="0x" + "22" * 32,
        agent_wallet_address="0x" + "b" * 40,
    )


def test_save_then_load_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "agent.json"
    session = _make_session()
    save_session(session, path=path)

    loaded = load_session(path)
    assert loaded == session


def test_save_uses_0600_permissions(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "agent.json"
    save_session(_make_session(), path=path)
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


def test_load_returns_none_when_absent(tmp_path: Path) -> None:
    assert load_session(tmp_path / "missing.json") is None


def test_load_raises_on_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("not json")
    with pytest.raises(SessionFileError):
        load_session(path)


def test_load_raises_when_top_level_is_not_object(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(SessionFileError):
        load_session(path)
