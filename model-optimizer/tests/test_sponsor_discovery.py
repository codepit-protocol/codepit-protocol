"""Tests for sponsor-competition discovery (slice G, #276).

The kit-side discovery helper lets an external agent find and enter a
*rewarded* sponsor competition instead of relying on bootstrap luck. It
reads the public challenge list, filters to open sponsor competitions on
the agent's lane, ranks them by reward pool, and confirms eligibility via
the authoritative ``/v1/agents/:id/eligibility`` endpoint before targeting.
"""

from __future__ import annotations

from typing import Any

from codepit_optimizer.sponsor_discovery import (
    discover_sponsor_challenge,
    rank_sponsor_challenges,
)

LANE = "ollama-gguf-local"


def _sponsor_challenge(
    challenge_id: str,
    *,
    pool_raw: str,
    lane: str = LANE,
    lifecycle_state: str = "Open",
    sponsor_competition: bool = True,
) -> dict[str, Any]:
    return {
        "challenge_id": challenge_id,
        "lifecycle_state": lifecycle_state,
        "artifact_lane": lane,
        "sponsor_competition": sponsor_competition,
        "bounty_terms": {"total_pool_raw": pool_raw, "funding_source": "sponsor"},
    }


def test_ranks_open_sponsor_challenges_by_reward_pool_descending() -> None:
    items = [
        _sponsor_challenge("small", pool_raw="100"),
        _sponsor_challenge("big", pool_raw="900"),
        _sponsor_challenge("medium", pool_raw="500"),
    ]

    ranked = rank_sponsor_challenges(items, artifact_lane=LANE)

    assert [c["challenge_id"] for c in ranked] == ["big", "medium", "small"]


def test_excludes_non_sponsor_closed_wrong_lane_and_unfunded() -> None:
    items = [
        _sponsor_challenge("ok", pool_raw="100"),
        _sponsor_challenge("not-sponsor", pool_raw="999", sponsor_competition=False),
        _sponsor_challenge("closed", pool_raw="999", lifecycle_state="Closed"),
        _sponsor_challenge("wrong-lane", pool_raw="999", lane="onnx-browser-webgpu"),
        _sponsor_challenge("zero-pool", pool_raw="0"),
    ]

    ranked = rank_sponsor_challenges(items, artifact_lane=LANE)

    # only the genuinely enterable, funded sponsor competition survives
    assert [c["challenge_id"] for c in ranked] == ["ok"]


def test_missing_bounty_terms_is_treated_as_unfunded() -> None:
    item = {
        "challenge_id": "no-terms",
        "lifecycle_state": "Open",
        "artifact_lane": LANE,
        "sponsor_competition": True,
        # no bounty_terms at all
    }

    assert rank_sponsor_challenges([item], artifact_lane=LANE) == []


class _StubClient:
    """Minimal CodePitClient stand-in for discovery: records eligibility probes."""

    def __init__(self, items: list[dict[str, Any]], eligible_ids: set[str]) -> None:
        self._items = items
        self._eligible_ids = eligible_ids
        self.eligibility_calls: list[str] = []

    def list_public_challenges(self) -> dict[str, Any]:
        return {"items": self._items}

    def read_eligibility(self, challenge_id: str) -> dict[str, Any]:
        self.eligibility_calls.append(challenge_id)
        eligible = challenge_id in self._eligible_ids
        return {"eligible": eligible, "reasons": [] if eligible else ["capability_mismatch"]}


def test_discover_returns_richest_eligible_skipping_ineligible_higher_pool() -> None:
    items = [
        _sponsor_challenge("rich-ineligible", pool_raw="900"),
        _sponsor_challenge("mid-eligible", pool_raw="500"),
        _sponsor_challenge("low-eligible", pool_raw="100"),
    ]
    client = _StubClient(items, eligible_ids={"mid-eligible", "low-eligible"})

    chosen = discover_sponsor_challenge(client, artifact_lane=LANE)

    # richest pool wins among *eligible* challenges; the richer ineligible one
    # is probed first (and skipped), the richest eligible one is targeted
    assert chosen == "mid-eligible"
    assert client.eligibility_calls == ["rich-ineligible", "mid-eligible"]


def test_discover_returns_none_when_no_eligible_sponsor_challenge() -> None:
    items = [_sponsor_challenge("rich", pool_raw="900")]
    client = _StubClient(items, eligible_ids=set())

    assert discover_sponsor_challenge(client, artifact_lane=LANE) is None


def test_discover_returns_none_when_list_has_no_sponsor_challenges() -> None:
    items = [_sponsor_challenge("bootstrap", pool_raw="0", sponsor_competition=False)]
    client = _StubClient(items, eligible_ids={"bootstrap"})

    assert discover_sponsor_challenge(client, artifact_lane=LANE) is None
    # no eligibility calls wasted on structurally non-targetable challenges
    assert client.eligibility_calls == []


# --------------------------------------------------------------------------
# Winnability filter (#291): skip challenges whose baseline the agent's
# allowed export class cannot beat. Falls back to pool-ranking when the
# baseline_optimization_methods signal is absent (no regression).
# --------------------------------------------------------------------------


def _sponsor_with_baseline(
    challenge_id: str,
    *,
    pool_raw: str,
    baseline_optimization_methods: list[str] | None,
) -> dict[str, Any]:
    item = _sponsor_challenge(challenge_id, pool_raw=pool_raw)
    if baseline_optimization_methods is not None:
        item["baseline_optimization_methods"] = baseline_optimization_methods
    return item


def test_discover_prefers_winnable_over_richer_unwinnable() -> None:
    # richer pool but baseline is already q4_k_m -> a q4_k_m export cannot beat it
    items = [
        _sponsor_with_baseline("rich-unwinnable", pool_raw="900", baseline_optimization_methods=["q4_k_m"]),
        _sponsor_with_baseline("poor-winnable", pool_raw="100", baseline_optimization_methods=["none"]),
    ]
    client = _StubClient(items, eligible_ids={"rich-unwinnable", "poor-winnable"})

    chosen = discover_sponsor_challenge(client, artifact_lane=LANE, allowed_export_target="gguf-q4-k-m")

    assert chosen == "poor-winnable"
    # the unwinnable richer challenge is skipped before any eligibility probe
    assert client.eligibility_calls == ["poor-winnable"]


def test_discover_returns_none_when_all_candidates_unwinnable() -> None:
    items = [
        _sponsor_with_baseline("a", pool_raw="900", baseline_optimization_methods=["q4_k_m"]),
        _sponsor_with_baseline("b", pool_raw="100", baseline_optimization_methods=["imatrix-q4_k_m"]),
    ]
    client = _StubClient(items, eligible_ids={"a", "b"})

    assert discover_sponsor_challenge(client, artifact_lane=LANE, allowed_export_target="gguf-q4-k-m") is None
    assert client.eligibility_calls == []


def test_discover_without_export_target_keeps_pool_ranking() -> None:
    # no allowed_export_target supplied -> winnability not enforced (back-compat)
    items = [
        _sponsor_with_baseline("rich", pool_raw="900", baseline_optimization_methods=["q4_k_m"]),
        _sponsor_with_baseline("poor", pool_raw="100", baseline_optimization_methods=["none"]),
    ]
    client = _StubClient(items, eligible_ids={"rich", "poor"})

    assert discover_sponsor_challenge(client, artifact_lane=LANE) == "rich"


def test_discover_with_export_target_but_no_baseline_signal_keeps_pool_ranking() -> None:
    # baseline_optimization_methods absent on items -> cannot assess, don't skip
    items = [
        _sponsor_challenge("rich", pool_raw="900"),
        _sponsor_challenge("poor", pool_raw="100"),
    ]
    client = _StubClient(items, eligible_ids={"rich", "poor"})

    assert discover_sponsor_challenge(client, artifact_lane=LANE, allowed_export_target="gguf-q4-k-m") == "rich"
