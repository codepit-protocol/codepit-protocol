"""Issue #258: register-external must declare the live tiny-chat lane by default.

A fresh external agent following the documented zero-human path
(`register-external` -> `modelbook-run --submit`) previously registered with the
ONNX/encoder lane and hit `capability_mismatch` against the only open challenge.
These deterministic tests pin the default lane to tiny-chat (the live
`ollama-gguf-local` / `chat-causal-small` lane) and lock the lane resolver.
"""

import pytest

from codepit_optimizer.orchestrator import (
    DEFAULT_REGISTER_LANE,
    REGISTER_LANES,
    resolve_register_lane_capabilities,
)
from codepit_optimizer.tiny_chat_bundle import OLLAMA_GGUF_LOCAL_ARTIFACT_LANE


def test_default_register_lane_is_tiny_chat():
    assert DEFAULT_REGISTER_LANE == "tiny-chat"
    assert "tiny-chat" in REGISTER_LANES


def test_default_lane_matches_live_challenge_admission_rules():
    caps = resolve_register_lane_capabilities()  # default lane
    # Must match the open challenge's admission rules so /v1/challenges/next is
    # eligible (model_class chat-causal-small, artifact_lane ollama-gguf-local).
    assert caps["declared_model_classes"] == ["chat-causal-small"]
    assert caps["declared_artifact_lanes"] == [OLLAMA_GGUF_LOCAL_ARTIFACT_LANE]
    assert caps["declared_artifact_lanes"] == ["ollama-gguf-local"]


def test_onnx_encoder_lane_preserved_for_targeted_callers():
    caps = resolve_register_lane_capabilities("onnx-encoder")
    assert caps["declared_model_classes"] == ["encoder-text-small"]
    assert caps["declared_artifact_lanes"] == ["onnx-browser-webgpu"]


def test_declared_at_version_threaded_into_capabilities():
    caps = resolve_register_lane_capabilities("tiny-chat", declared_at_version="v9")
    assert caps["declared_at_version"] == "v9"


def test_unknown_lane_raises():
    with pytest.raises(ValueError, match="unknown registration lane"):
        resolve_register_lane_capabilities("does-not-exist")
