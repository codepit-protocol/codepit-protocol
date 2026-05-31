"""Wire-shape tests for the Modelbook iteration.

The engine has its own integration tests for the modelbook routes themselves.
This file pins the agent-side contract: discovery, run creation, decisions,
real Tiny Chat packaging, artifact registration, and fail-closed behavior.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
from urllib.parse import unquote, urlparse

import httpx
import pytest

from codepit_optimizer.modelbook_loop import (
    ModelbookIterationConfig,
    ModelbookLoopError,
    run_modelbook_iteration,
    run_modelbook_loop,
)
from codepit_optimizer.protocol import CodePitClient
from codepit_optimizer.tiny_chat_packager import TinyChatPackagingError


def _build_handler(
    *,
    available_items: list[dict],
    context: dict,
    run_id: str = "run_1",
    artifact_set_id: str = "art_1",
    captures: list[httpx.Request],
) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        captures.append(request)
        path = request.url.path
        if path == "/v2/modelbooks/available" and request.method == "GET":
            return httpx.Response(200, json={"items": available_items})
        if path.startswith("/v2/modelbooks/") and path.endswith("/context"):
            return httpx.Response(200, json=context)
        if path.startswith("/v2/modelbooks/") and path.endswith("/runs"):
            return httpx.Response(201, json={"run": {"training_run_id": run_id}})
        if path.startswith("/v2/runs/") and path.endswith("/decisions"):
            return httpx.Response(201, json={"decision": {"decision_id": "dec_x"}})
        if path.startswith("/v2/runs/") and path.endswith("/events"):
            return httpx.Response(201, json={"event": {"event_id": "evt_x"}})
        if path.startswith("/v2/modelbooks/") and path.endswith("/posts"):
            return httpx.Response(201, json={"post": {"modelbook_post_id": "post_x"}})
        if path.startswith("/v2/runs/") and path.endswith("/artifacts"):
            return httpx.Response(
                201,
                json={"artifact_set": {"artifact_set_id": artifact_set_id}},
            )
        return httpx.Response(404, json={"error": {"code": "not_handled", "path": path}})

    return handler


def _make_client(handler: Callable[[httpx.Request], httpx.Response]) -> CodePitClient:
    return CodePitClient(
        "http://engine.test/",
        agent_id="agent_1",
        credential="bearer-secret",
        transport=httpx.MockTransport(handler),
    )


def _path_from_file_ref(ref: str) -> Path:
    parsed = urlparse(ref)
    assert parsed.scheme == "file"
    return Path(unquote(parsed.path))


def test_iteration_hits_full_modelbook_pipeline_in_order(tmp_path: Path) -> None:
    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "status": "active",
                "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                "artifact_lane": "ollama-gguf-local",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1"},
            "assigned_agent": {"agent_id": "agent_1"},
            "policy": {
                "allowed_training_methods": ["lora", "qlora"],
                "allowed_export_targets": ["gguf-q4-k-m"],
                "allowed_dataset_shard_ids": [],
            },
        },
        captures=captures,
    )
    client = _make_client(handler)

    result = run_modelbook_iteration(
        client,
        ModelbookIterationConfig(artifact_output_dir=tmp_path),
    )

    assert result.skipped_reason is None
    assert result.modelbook_id == "mb_1"
    assert result.training_run_id == "run_1"
    assert result.recipe_kind == "lora"
    assert result.decisions_recorded == 3  # recipe, hyperparameters, export
    assert result.events_emitted == 1 + 1 + 1 + 3 + 1 + 1  # implicit + feed/start/progress/complete/feed
    assert result.artifact_set_id == "art_1"
    assert result.stub_training_used is False

    # Order must be: discover -> context -> create run -> decisions + events + artifact + feed.
    # interleaved as built by run_modelbook_iteration.
    methods_paths = [(r.method, r.url.path) for r in captures]
    assert methods_paths[0] == ("GET", "/v2/modelbooks/available")
    assert methods_paths[1] == ("GET", "/v2/modelbooks/mb_1/context")
    assert methods_paths[2] == ("POST", "/v2/modelbooks/mb_1/runs")
    remainder = methods_paths[3:]
    assert all(path.startswith("/v2/runs/run_1/") for _, path in remainder)
    assert remainder[:3] == [
        ("POST", "/v2/runs/run_1/decisions"),
        ("POST", "/v2/runs/run_1/decisions"),
        ("POST", "/v2/runs/run_1/decisions"),
    ]
    assert ("POST", "/v2/runs/run_1/artifacts") in remainder
    assert remainder[-1] == ("POST", "/v2/runs/run_1/events")


def test_iteration_autonomously_submits_package_to_canonical_submission_path(
    tmp_path: Path,
) -> None:
    captures: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captures.append(request)
        path = request.url.path
        if path == "/v2/modelbooks/available" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "modelbook_id": "mb_1",
                            "status": "active",
                            "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                            "model_class": "chat-causal-small",
                            "artifact_lane": "ollama-gguf-local",
                        }
                    ]
                },
            )
        if path == "/v2/modelbooks/mb_1/context" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "modelbook": {
                        "modelbook_id": "mb_1",
                        "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                        "model_class": "chat-causal-small",
                        "artifact_lane": "ollama-gguf-local",
                    },
                    "assigned_agent": {"agent_id": "agent_1"},
                    "policy": {
                        "allowed_training_methods": ["lora"],
                        "allowed_export_targets": ["gguf-q4-k-m"],
                        "allowed_dataset_shard_ids": [],
                    },
                    "dataset_shards": [],
                },
            )
        if path == "/v2/modelbooks/mb_1/runs" and request.method == "POST":
            return httpx.Response(201, json={"run": {"training_run_id": "run_1"}})
        if path.startswith("/v2/runs/run_1/") and path.endswith("/decisions"):
            return httpx.Response(201, json={"decision": {"decision_id": "dec_x"}})
        if path.startswith("/v2/runs/run_1/") and path.endswith("/events"):
            return httpx.Response(201, json={"event": {"event_id": "evt_x"}})
        if path == "/v2/runs/run_1/artifacts" and request.method == "POST":
            return httpx.Response(201, json={"artifact_set": {"artifact_set_id": "art_1"}})
        if path == "/v1/challenges/next" and request.method == "GET":
            return httpx.Response(
                200,
                json={"challenge": {"challenge_id": "ch_tiny"}},
            )
        if path == "/v1/challenges/ch_tiny" and request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "challenge_id": "ch_tiny",
                    "benchmark_target_version": "tiny-chat-v1",
                    "artifact_lane": "ollama-gguf-local",
                    "model_class_admission_rules": ["chat-causal-small"],
                    "lifecycle_state": "Open",
                },
            )
        if path == "/v1/submissions" and request.method == "POST":
            body = json.loads(request.content.decode())
            files = body["manifest_envelope"]["files"]
            return httpx.Response(
                201,
                json={
                    "submission_id": "sub_1",
                    "state": "UPLOADING",
                    "upload_orchestration": {
                        "kind": "presigned-urls",
                        "expires_at": "2099-01-01T00:00:00Z",
                        "files": [
                            {
                                "logical_name": file["logical_name"],
                                "role": file["role"],
                                "media_type": file["media_type"],
                                "size_bytes": file["size_bytes"],
                                "sha256": file["sha256"],
                                "object_key": f"submissions/sub_1/{file['logical_name']}",
                                "upload_url": f"http://engine.test/uploads/{file['logical_name']}",
                            }
                            for file in files
                        ],
                    },
                },
            )
        if path.startswith("/uploads/") and request.method == "PUT":
            return httpx.Response(200)
        if path == "/v2/runs/run_1/submit" and request.method == "POST":
            return httpx.Response(
                200,
                json={"run": {"training_run_id": "run_1", "submission_id": "sub_1"}},
            )
        return httpx.Response(404, json={"error": {"code": "not_handled", "path": path}})

    client = _make_client(handler)
    result = run_modelbook_iteration(
        client,
        ModelbookIterationConfig(artifact_output_dir=tmp_path, submit=True),
    )

    assert result.submission_id == "sub_1"
    assert result.submission_state == "UPLOADING"
    assert result.challenge_id == "ch_tiny"

    create_submission = next(req for req in captures if req.url.path == "/v1/submissions")
    body = json.loads(create_submission.content.decode())
    assert body["agent_id"] == "agent_1"
    assert body["challenge_id"] == "ch_tiny"
    assert body["manifest_envelope"]["artifact_lane"] == "ollama-gguf-local"
    assert body["manifest_envelope"]["runtime_target"]["environment_family"] == "local-ollama"
    logical_names = {file["logical_name"] for file in body["manifest_envelope"]["files"]}
    assert any(name.endswith(".gguf") for name in logical_names)
    assert {"Modelfile", "provenance.json", "checksums.json"} <= logical_names

    methods_paths = [(r.method, r.url.path) for r in captures]
    assert ("GET", "/v1/challenges/next") in methods_paths
    assert ("POST", "/v1/submissions") in methods_paths
    assert ("POST", "/v2/runs/run_1/submit") in methods_paths
    upload_requests = [req for req in captures if req.url.path.startswith("/uploads/")]
    assert len(upload_requests) == len(body["manifest_envelope"]["files"])
    assert all("authorization" not in req.headers for req in upload_requests)
    authenticated_paths = {
        "/v2/modelbooks/mb_1/context",
        "/v2/modelbooks/mb_1/runs",
        "/v1/challenges/next",
        "/v1/challenges/ch_tiny",
        "/v1/submissions",
        "/v2/runs/run_1/submit",
    }
    assert all(
        req.headers.get("authorization") == "Bearer bearer-secret"
        for req in captures
        if req.url.path in authenticated_paths
    )

    event_bodies = [
        json.loads(req.content.decode())
        for req in captures
        if req.url.path.endswith("/events")
    ]
    submit_posts = [
        body
        for body in event_bodies
        if body["event_type"] == "feed.agent_post"
        and body["metadata"].get("phase") == "submitted"
    ]
    assert len(submit_posts) == 1
    assert submit_posts[0]["metadata"]["submission_id"] == "sub_1"
    assert "platform benchmark" in submit_posts[0]["message"]


def test_artifact_set_request_respects_policy_and_uses_real_package(tmp_path: Path) -> None:
    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "status": "active",
                "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                "artifact_lane": "ollama-gguf-local",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1"},
            "assigned_agent": {"agent_id": "agent_1"},
            "policy": {
                "allowed_training_methods": ["lora"],
                "allowed_export_targets": ["gguf-q4-k-m"],
                "allowed_dataset_shard_ids": [],
            },
        },
        captures=captures,
    )
    client = _make_client(handler)
    result = run_modelbook_iteration(
        client, ModelbookIterationConfig(artifact_output_dir=tmp_path)
    )

    artifact_calls = [
        r for r in captures if r.url.path.endswith("/artifacts") and r.method == "POST"
    ]
    assert len(artifact_calls) == 1
    body = json.loads(artifact_calls[0].content.decode("utf-8"))
    assert body["artifact_lane"] == "ollama-gguf-local"
    assert body["quantization_profile"] == "gguf-q4-k-m"
    assert body["dataset_shard_ids"] == []
    assert body["primary_artifact_ref"].startswith("file://")
    assert body["adapter_ref"].startswith("file://")
    assert body["merged_model_ref"].startswith("file://")
    assert body["gguf_ref"].startswith("file://")
    assert body["modelfile_ref"].startswith("file://")
    assert body["checksum_ref"].startswith("file://")
    assert body["provenance"]["training_algorithm"] == "deterministic-token-frequency-adapter"
    assert "checksums" in body["provenance"]
    assert "stub" not in json.dumps(body)

    for ref in [
        body["primary_artifact_ref"],
        body["adapter_ref"],
        body["merged_model_ref"],
        body["gguf_ref"],
        body["modelfile_ref"],
        body["checksum_ref"],
    ]:
        assert _path_from_file_ref(ref).exists()

    event_bodies = [
        json.loads(req.content.decode())
        for req in captures
        if req.url.path.endswith("/events")
    ]
    assert event_bodies
    assert not any((body.get("metadata") or {}).get("stub") is True for body in event_bodies)
    assert {body["event_type"] for body in event_bodies} >= {
        "feed.agent_post",
        "training.dataset_prepared",
        "training.adapter_trained",
        "training.packaged",
        "training.complete",
    }
    feed_posts = [
        body for body in event_bodies if body["event_type"] == "feed.agent_post"
    ]
    assert len(feed_posts) >= 2
    assert feed_posts[0]["metadata"]["title"] == "Training plan chosen"
    assert "small local model" in feed_posts[0]["message"]
    assert feed_posts[-1]["metadata"]["title"] == "Model package ready"
    assert result.artifact_set_id == feed_posts[-1]["metadata"]["artifact_set_id"]


def test_packaging_failure_records_failure_event_without_artifact_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "status": "active",
                "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                "artifact_lane": "ollama-gguf-local",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1"},
            "assigned_agent": {"agent_id": "agent_1"},
            "policy": {
                "allowed_training_methods": ["lora"],
                "allowed_export_targets": ["gguf-q4-k-m"],
                "allowed_dataset_shard_ids": [],
            },
        },
        captures=captures,
    )
    client = _make_client(handler)

    def fail_package(**_kwargs):
        raise TinyChatPackagingError("checksum generation unavailable")

    monkeypatch.setattr(
        "codepit_optimizer.modelbook_loop.train_and_package_tiny_chat",
        fail_package,
    )

    with pytest.raises(ModelbookLoopError, match="tiny-chat training/package failed"):
        run_modelbook_iteration(client, ModelbookIterationConfig(artifact_output_dir=tmp_path))

    assert not any(req.url.path.endswith("/artifacts") for req in captures)
    event_bodies = [
        json.loads(req.content.decode())
        for req in captures
        if req.url.path.endswith("/events")
    ]
    assert any(body["event_type"] == "training.failed" for body in event_bodies)
    assert not any((body.get("metadata") or {}).get("stub") is True for body in event_bodies)


def test_iteration_skips_when_no_modelbook_available() -> None:
    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[],
        context={},
        captures=captures,
    )
    client = _make_client(handler)

    result = run_modelbook_iteration(client)

    assert result.skipped_reason == "no_available_modelbook"
    assert result.training_run_id is None
    assert result.artifact_set_id is None
    # Only the discovery call was made.
    assert [(r.method, r.url.path) for r in captures] == [
        ("GET", "/v2/modelbooks/available")
    ]


def test_pinned_recipe_must_be_in_allowed_methods() -> None:
    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "status": "active",
                "artifact_lane": "ollama-gguf-local",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1"},
            "assigned_agent": {"agent_id": "agent_1"},
            "policy": {
                "allowed_training_methods": ["lora"],
                "allowed_export_targets": ["gguf-q4-k-m"],
                "allowed_dataset_shard_ids": [],
            },
        },
        captures=captures,
    )
    client = _make_client(handler)

    with pytest.raises(ModelbookLoopError):
        run_modelbook_iteration(
            client,
            ModelbookIterationConfig(recipe_kind="full-finetune"),
        )


def test_loop_sleeps_on_skip_then_returns_after_max_iterations(tmp_path: Path) -> None:
    captures: list[httpx.Request] = []
    available: list[dict] = []

    def staged_handler(request: httpx.Request) -> httpx.Response:
        captures.append(request)
        if request.url.path == "/v2/modelbooks/available" and request.method == "GET":
            return httpx.Response(200, json={"items": list(available)})
        return _build_handler(
            available_items=available,
            context={
                "modelbook": {"modelbook_id": "mb_1"},
                "assigned_agent": {"agent_id": "agent_1"},
                "policy": {
                    "allowed_training_methods": ["lora"],
                    "allowed_export_targets": ["gguf-q4-k-m"],
                    "allowed_dataset_shard_ids": [],
                },
            },
            captures=captures,
        )(request)

    client = _make_client(staged_handler)

    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        # On the second sleep we publish a modelbook so the next iteration
        # completes and `max_iterations=1` returns.
        if len(sleeps) == 1:
            available.append(
                {
                    "modelbook_id": "mb_1",
                    "status": "active",
                    "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                    "artifact_lane": "ollama-gguf-local",
                }
            )

    results = run_modelbook_loop(
        client,
        ModelbookIterationConfig(artifact_output_dir=tmp_path),
        max_iterations=1,
        idle_sleep_seconds=0.25,
        sleep=fake_sleep,
    )

    assert len(results) == 2
    assert results[0].skipped_reason == "no_available_modelbook"
    assert results[1].skipped_reason is None
    assert results[1].training_run_id == "run_1"
    assert sleeps == [0.25]


# --------------------------------------------------------------------------
# Brain-driven decisions (real LLM path)
# --------------------------------------------------------------------------


class _FakeBrain:
    """Stand-in for ManagedBrainProvider that returns canned JSON per action.

    Tests express the canned LLM output keyed by action_id. Each call also
    records the prompt/system/tier/schema for assertion.
    """

    def __init__(self, responses: dict[str, dict]) -> None:
        # responses[action_id] = {"content": str, "provider": str, "model": str}
        self._responses = responses
        self.calls: list[dict] = []

    def generate_with_metadata(
        self,
        *,
        prompt,
        action_id,
        attempt,
        tier,
        schema=None,
        system=None,
    ):
        from codepit_optimizer.brain_providers.managed import ManagedBrainResponse

        self.calls.append(
            {
                "action_id": action_id,
                "tier": tier,
                "attempt": attempt,
                "schema": dict(schema) if schema else None,
                "system": system,
                "prompt": prompt,
            }
        )
        if action_id not in self._responses:
            raise AssertionError(f"_FakeBrain missing canned response for {action_id!r}")
        canned = self._responses[action_id]
        return ManagedBrainResponse(
            content=canned["content"],
            provider=canned.get("provider"),
            model=canned.get("model"),
            tier=tier,
        )


def _brain_responses_with_social(social_content: str) -> dict[str, dict]:
    return {
        "modelbook-recipe": {
            "content": json.dumps(
                {
                    "recipe_kind": "lora",
                    "rationale": "LoRA keeps the training step lightweight.",
                    "risks": [],
                }
            ),
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
        },
        "modelbook-hyperparams": {
            "content": json.dumps(
                {
                    "learning_rate": 1e-4,
                    "epochs": 1,
                    "rank": 8,
                    "alpha": 16,
                    "dropout": 0.05,
                    "rationale": "Conservative defaults for a tiny chat model.",
                }
            ),
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
        },
        "modelbook-export": {
            "content": json.dumps(
                {
                    "quantization_profile": "gguf-q4-k-m",
                    "rationale": "Q4 keeps the local artifact small.",
                }
            ),
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
        },
        "modelbook-social": {
            "content": social_content,
            "provider": "groq",
            "model": "llama-3.1-8b-instant",
        },
    }


def test_social_step_posts_first_class_update_when_enabled(tmp_path: Path) -> None:
    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "display_name": "CodePit Tiny Chat",
                "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                "artifact_lane": "ollama-gguf-local",
                "status": "active",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1", "display_name": "CodePit Tiny Chat"},
            "assigned_agent": {"agent_id": "agent_1"},
            "policy": {
                "allowed_training_methods": ["lora"],
                "allowed_export_targets": ["gguf-q4-k-m"],
                "allowed_dataset_shard_ids": [],
            },
        },
        captures=captures,
    )
    client = _make_client(handler)
    brain = _FakeBrain(
        _brain_responses_with_social(
            json.dumps(
                {
                    "action": "post",
                    "title": "Tiny Chat package plan",
                    "body": "I picked a compact local package so the model can run on laptop hardware.",
                }
            )
        )
    )

    result = run_modelbook_iteration(
        client,
        ModelbookIterationConfig(
            brain=brain,
            social_posts_enabled=True,
            artifact_output_dir=tmp_path,
        ),
    )

    post_requests = [req for req in captures if req.url.path == "/v2/modelbooks/mb_1/posts"]
    assert len(post_requests) == 1
    body = json.loads(post_requests[0].content.decode())
    assert body["training_run_id"] == "run_1"
    assert body["client_post_id"] == "run_1:social:1"
    assert body["title"] == "Tiny Chat package plan"
    assert "laptop hardware" in body["body"]
    assert "parent_post_id" not in body
    assert result.social_posts_created == 1
    assert result.social_post_failures == 0


def test_social_step_can_reply_to_parent_post(tmp_path: Path) -> None:
    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                "artifact_lane": "ollama-gguf-local",
                "status": "active",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1"},
            "assigned_agent": {"agent_id": "agent_1"},
            "policy": {
                "allowed_training_methods": ["lora"],
                "allowed_export_targets": ["gguf-q4-k-m"],
                "allowed_dataset_shard_ids": [],
            },
        },
        captures=captures,
    )
    client = _make_client(handler)
    brain = _FakeBrain(
        _brain_responses_with_social(
            json.dumps(
                {
                    "action": "reply",
                    "parent_post_id": "post_parent",
                    "body": "That is why small local chat models are useful for support teams.",
                }
            )
        )
    )

    result = run_modelbook_iteration(
        client,
        ModelbookIterationConfig(
            brain=brain,
            social_posts_enabled=True,
            artifact_output_dir=tmp_path,
        ),
    )

    post_request = next(req for req in captures if req.url.path == "/v2/modelbooks/mb_1/posts")
    body = json.loads(post_request.content.decode())
    assert body["parent_post_id"] == "post_parent"
    assert "support teams" in body["body"]
    assert result.social_posts_created == 1


def test_social_step_silent_action_does_not_create_post(tmp_path: Path) -> None:
    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                "artifact_lane": "ollama-gguf-local",
                "status": "active",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1"},
            "assigned_agent": {"agent_id": "agent_1"},
            "policy": {
                "allowed_training_methods": ["lora"],
                "allowed_export_targets": ["gguf-q4-k-m"],
                "allowed_dataset_shard_ids": [],
            },
        },
        captures=captures,
    )
    client = _make_client(handler)
    brain = _FakeBrain(_brain_responses_with_social(json.dumps({"action": "silent"})))

    result = run_modelbook_iteration(
        client,
        ModelbookIterationConfig(
            brain=brain,
            social_posts_enabled=True,
            artifact_output_dir=tmp_path,
        ),
    )

    assert not any(req.url.path == "/v2/modelbooks/mb_1/posts" for req in captures)
    assert result.social_posts_created == 0
    assert result.social_post_failures == 0


def test_social_step_malformed_brain_output_is_failure_isolated(tmp_path: Path) -> None:
    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                "artifact_lane": "ollama-gguf-local",
                "status": "active",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1"},
            "assigned_agent": {"agent_id": "agent_1"},
            "policy": {
                "allowed_training_methods": ["lora"],
                "allowed_export_targets": ["gguf-q4-k-m"],
                "allowed_dataset_shard_ids": [],
            },
        },
        captures=captures,
    )
    client = _make_client(handler)
    brain = _FakeBrain(_brain_responses_with_social("not json"))

    result = run_modelbook_iteration(
        client,
        ModelbookIterationConfig(
            brain=brain,
            social_posts_enabled=True,
            artifact_output_dir=tmp_path,
        ),
    )

    assert result.artifact_set_id == "art_1"
    assert result.social_posts_created == 0
    assert result.social_post_failures == 1
    assert not any(req.url.path == "/v2/modelbooks/mb_1/posts" for req in captures)


def test_social_step_http_failure_is_failure_isolated(tmp_path: Path) -> None:
    captures: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/modelbooks/mb_1/posts":
            captures.append(request)
            return httpx.Response(503, json={"error": {"code": "unavailable"}})
        if request.url.path == "/v1/challenges/next" and request.method == "GET":
            captures.append(request)
            return httpx.Response(200, json={"challenge": {"challenge_id": "ch_tiny"}})
        if request.url.path == "/v1/challenges/ch_tiny" and request.method == "GET":
            captures.append(request)
            return httpx.Response(
                200,
                json={
                    "challenge_id": "ch_tiny",
                    "benchmark_target_version": "tiny-chat-v1",
                    "artifact_lane": "ollama-gguf-local",
                    "model_class_admission_rules": ["chat-causal-small"],
                    "lifecycle_state": "Open",
                },
            )
        if request.url.path == "/v1/submissions" and request.method == "POST":
            captures.append(request)
            body = json.loads(request.content.decode())
            files = body["manifest_envelope"]["files"]
            return httpx.Response(
                201,
                json={
                    "submission_id": "sub_after_social_failure",
                    "state": "UPLOADING",
                    "upload_orchestration": {
                        "kind": "presigned-urls",
                        "expires_at": "2099-01-01T00:00:00Z",
                        "files": [
                            {
                                "logical_name": file["logical_name"],
                                "role": file["role"],
                                "media_type": file["media_type"],
                                "size_bytes": file["size_bytes"],
                                "sha256": file["sha256"],
                                "object_key": f"submissions/sub_after_social_failure/{file['logical_name']}",
                                "upload_url": f"http://engine.test/uploads/{file['logical_name']}",
                            }
                            for file in files
                        ],
                    },
                },
            )
        if request.url.path.startswith("/uploads/") and request.method == "PUT":
            captures.append(request)
            return httpx.Response(200)
        if request.url.path == "/v2/runs/run_1/submit" and request.method == "POST":
            captures.append(request)
            return httpx.Response(
                200,
                json={
                    "run": {
                        "training_run_id": "run_1",
                        "submission_id": "sub_after_social_failure",
                    }
                },
            )
        return _build_handler(
            available_items=[
                {
                    "modelbook_id": "mb_1",
                    "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                    "artifact_lane": "ollama-gguf-local",
                    "status": "active",
                }
            ],
            context={
                "modelbook": {"modelbook_id": "mb_1"},
                "assigned_agent": {"agent_id": "agent_1"},
                "policy": {
                    "allowed_training_methods": ["lora"],
                    "allowed_export_targets": ["gguf-q4-k-m"],
                    "allowed_dataset_shard_ids": [],
                },
            },
            captures=captures,
        )(request)

    client = _make_client(handler)
    brain = _FakeBrain(
        _brain_responses_with_social(
            json.dumps({"action": "post", "body": "This post endpoint is unavailable."})
        )
    )

    result = run_modelbook_iteration(
        client,
        ModelbookIterationConfig(
            brain=brain,
            social_posts_enabled=True,
            submit=True,
            artifact_output_dir=tmp_path,
        ),
    )

    assert result.artifact_set_id == "art_1"
    assert result.submission_id == "sub_after_social_failure"
    assert result.submission_state == "UPLOADING"
    assert result.challenge_id == "ch_tiny"
    assert result.social_posts_created == 0
    assert result.social_post_failures == 1
    assert any(req.url.path == "/v1/submissions" for req in captures)
    assert any(req.url.path == "/v2/runs/run_1/submit" for req in captures)


def test_iteration_uses_brain_for_every_decision_when_provided(tmp_path: Path):
    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "display_name": "CodePit Tiny Chat",
                "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                "model_class": "chat-causal-small",
                "artifact_lane": "ollama-gguf-local",
                "status": "active",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1"},
            "policy": {
                "allowed_training_methods": ["lora", "qlora"],
                "allowed_export_targets": ["q4_k_m", "q5_k_m"],
                "max_budget_codepit": "1000",
                "requires_publish_approval": True,
            },
            "dataset_shards": [],
        },
        captures=captures,
    )
    client = _make_client(handler)
    brain = _FakeBrain(
        {
            "modelbook-recipe": {
                "content": json.dumps(
                    {
                        "recipe_kind": "qlora",
                        "rationale": "QLoRA reduces VRAM and is appropriate for Qwen 0.5B.",
                        "risks": ["May undertrain at rank 4"],
                    }
                ),
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            },
            "modelbook-hyperparams": {
                "content": json.dumps(
                    {
                        "learning_rate": 2e-4,
                        "epochs": 2,
                        "rank": 16,
                        "alpha": 32,
                        "dropout": 0.05,
                        "bits": 4,
                        "rationale": "Slightly higher rank for chat fine-tune; epochs capped at 2.",
                    }
                ),
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            },
            "modelbook-export": {
                "content": json.dumps(
                    {
                        "quantization_profile": "q5_k_m",
                        "rationale": "Balanced quality/size for Ollama local use.",
                    }
                ),
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            },
        }
    )

    result = run_modelbook_iteration(
        client,
        ModelbookIterationConfig(
            brain=brain,
            brain_tier="cheap",
            artifact_output_dir=tmp_path,
        ),
    )

    # Result surfaces brain attribution
    assert result.brain_driven is True
    assert result.brain_provider == "groq"
    assert result.brain_model == "llama-3.1-8b-instant"
    assert result.recipe_kind == "qlora"  # came from LLM, not heuristic default
    assert result.decisions_recorded == 3

    # Brain was called exactly three times — one per decision — with correct tier
    assert [c["action_id"] for c in brain.calls] == [
        "modelbook-recipe",
        "modelbook-hyperparams",
        "modelbook-export",
    ]
    for call in brain.calls:
        assert call["tier"] == "cheap"
        # Schema is embedded in the prompt body, not sent as response_format,
        # because Groq's smaller models reject json_schema mode.
        assert call["schema"] is None
        assert "Return JSON only" in call["prompt"]
        assert "JSON" in (call["system"] or "")  # system prompt enforces JSON

    # Every decision request body carries the real brain provider/model
    decision_bodies = [
        json.loads(req.content.decode())
        for req in captures
        if req.url.path.endswith("/decisions")
    ]
    assert len(decision_bodies) == 3
    for body in decision_bodies:
        assert body["brain_provider"] == "groq"
        assert body["brain_model"] == "llama-3.1-8b-instant"

    recipe_body = decision_bodies[0]
    assert recipe_body["decision_type"] == "recipe"
    assert recipe_body["selected_inputs"]["recipe_kind"] == "qlora"
    assert "VRAM" in recipe_body["rationale"]
    assert recipe_body["risk_notes"] == ["May undertrain at rank 4"]
    assert "lora" in recipe_body["rejected_options"]

    hp_body = decision_bodies[1]
    assert hp_body["decision_type"] == "hyperparameters"
    assert hp_body["selected_inputs"]["learning_rate"] == 2e-4
    assert hp_body["selected_inputs"]["bits"] == 4
    assert "rationale" not in hp_body["selected_inputs"]  # rationale lives outside

    export_body = decision_bodies[2]
    assert export_body["decision_type"] == "export"
    assert export_body["selected_inputs"]["quantization_profile"] == "q5_k_m"


def test_brain_response_outside_policy_raises_brain_call_failed():
    from codepit_optimizer.modelbook_loop import BrainCallFailed

    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "display_name": "X",
                "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                "model_class": "chat-causal-small",
                "artifact_lane": "ollama-gguf-local",
                "status": "active",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1"},
            "policy": {
                "allowed_training_methods": ["lora"],
                "allowed_export_targets": ["q4_k_m"],
            },
        },
        captures=captures,
    )
    client = _make_client(handler)
    brain = _FakeBrain(
        {
            "modelbook-recipe": {
                "content": json.dumps(
                    {"recipe_kind": "full-finetune", "rationale": "ignoring policy"}
                ),
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            }
        }
    )

    with pytest.raises(BrainCallFailed, match="did not match any allowed value"):
        run_modelbook_iteration(client, ModelbookIterationConfig(brain=brain))


def test_brain_response_non_json_raises_brain_call_failed():
    from codepit_optimizer.modelbook_loop import BrainCallFailed

    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "display_name": "X",
                "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                "model_class": "chat-causal-small",
                "artifact_lane": "ollama-gguf-local",
                "status": "active",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1"},
            "policy": {
                "allowed_training_methods": ["lora"],
                "allowed_export_targets": ["q4_k_m"],
            },
        },
        captures=captures,
    )
    client = _make_client(handler)
    brain = _FakeBrain(
        {
            "modelbook-recipe": {
                "content": "I am an LLM and I refuse to respond in JSON. Here is some prose.",
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            }
        }
    )

    with pytest.raises(BrainCallFailed, match="not valid JSON"):
        run_modelbook_iteration(client, ModelbookIterationConfig(brain=brain))


def test_brain_response_handles_markdown_fenced_json(tmp_path: Path):
    """LLMs often wrap JSON in ```json ... ``` fences. We strip those."""
    captures: list[httpx.Request] = []
    handler = _build_handler(
        available_items=[
            {
                "modelbook_id": "mb_1",
                "display_name": "X",
                "base_model_ref": "hf://Qwen/Qwen2.5-0.5B-Instruct",
                "model_class": "chat-causal-small",
                "artifact_lane": "ollama-gguf-local",
                "status": "active",
            }
        ],
        context={
            "modelbook": {"modelbook_id": "mb_1"},
            "policy": {
                "allowed_training_methods": ["lora"],
                "allowed_export_targets": ["q4_k_m"],
            },
        },
        captures=captures,
    )
    client = _make_client(handler)
    brain = _FakeBrain(
        {
            "modelbook-recipe": {
                "content": '```json\n{"recipe_kind": "lora", "rationale": "ok"}\n```',
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            },
            "modelbook-hyperparams": {
                "content": '{"learning_rate": 1e-4, "epochs": 1, "rationale": "ok"}',
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            },
            "modelbook-export": {
                "content": '{"quantization_profile": "q4_k_m", "rationale": "ok"}',
                "provider": "groq",
                "model": "llama-3.1-8b-instant",
            },
        }
    )

    result = run_modelbook_iteration(
        client,
        ModelbookIterationConfig(brain=brain, artifact_output_dir=tmp_path),
    )
    assert result.recipe_kind == "lora"
    assert result.brain_provider == "groq"
