"""Sponsor-competition discovery for external agents (slice G, #276).

A freshly-joined external agent should be able to deliberately enter a
*rewarded* sponsor competition instead of getting a bootstrap challenge by
luck. This module reads the public challenge list, filters to open sponsor
competitions on the agent's artifact lane, ranks them by reward pool, and
(via :func:`discover_sponsor_challenge`) confirms eligibility against the
authoritative ``/v1/agents/:id/eligibility`` endpoint before targeting one.

Eligibility is intentionally delegated to the engine rather than
re-derived from list fields: trust-tier ordering and model-class admission
are engine-owned invariants, and duplicating them in the kit would drift.
The structural pre-filter here (sponsor + open + lane + funded) only avoids
spending eligibility calls on challenges that can never match.
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol, Sequence

#: Challenges are only enterable while Open (Section 1 §4.3.2 lifecycle).
_OPEN_LIFECYCLE_STATE = "Open"


def _normalize_quant_class(token: str) -> str:
    """Canonical quantization class for an export-target / optimization-method.

    Mirrors the engine guard ``baseline-beatability.ts`` so the kit and engine
    agree on what "same class" means:
      ``gguf-q4-k-m`` / ``Q4_K_M`` / ``imatrix-q4_k_m`` -> ``q4_k_m``
      ``f16`` / ``fp16`` / ``none`` / ``base`` -> ``""`` (unoptimized; headroom)
    """
    t = token.strip().lower()
    if not t:
        return ""
    if t.startswith("gguf-") or t.startswith("gguf_"):
        t = t[len("gguf-"):]
    if t.startswith("imatrix-") or t.startswith("imatrix_"):
        t = t[len("imatrix-"):]
    t = t.replace("-", "_")
    if t in {"f16", "fp16", "f32", "bf16", "none", "base"}:
        return ""
    return t


def _baseline_leaves_headroom(
    baseline_optimization_methods: Sequence[str],
    allowed_export_target: str,
) -> bool:
    """True if a submission in ``allowed_export_target`` could beat the baseline.

    The allowed export class leaves no headroom when the baseline is already an
    instance of it (e.g. q4_k_m baseline + q4_k_m export). An unoptimized
    (FP16) baseline always leaves headroom.
    """
    allowed_class = _normalize_quant_class(allowed_export_target)
    if not allowed_class:
        return True
    baseline_classes = {
        cls
        for cls in (_normalize_quant_class(m) for m in baseline_optimization_methods)
        if cls
    }
    if not baseline_classes:
        return True
    return allowed_class not in baseline_classes


def _pool_raw(challenge: Mapping[str, Any]) -> int:
    """Reward pool as an int in raw units; 0 when absent/unparseable."""
    terms = challenge.get("bounty_terms")
    if not isinstance(terms, Mapping):
        return 0
    raw = terms.get("total_pool_raw")
    try:
        return int(str(raw))
    except (TypeError, ValueError):
        return 0


def _is_targetable_sponsor(challenge: Mapping[str, Any], artifact_lane: str) -> bool:
    return (
        challenge.get("sponsor_competition") is True
        and challenge.get("lifecycle_state") == _OPEN_LIFECYCLE_STATE
        and challenge.get("artifact_lane") == artifact_lane
        and _pool_raw(challenge) > 0
    )


def rank_sponsor_challenges(
    items: Sequence[Mapping[str, Any]],
    *,
    artifact_lane: str,
) -> list[dict[str, Any]]:
    """Open sponsor competitions on ``artifact_lane``, richest pool first.

    Filters out non-sponsor, non-open, wrong-lane, and unfunded challenges,
    then sorts by reward pool descending. Ties keep input order (stable).
    """
    matches = [
        dict(challenge)
        for challenge in items
        if _is_targetable_sponsor(challenge, artifact_lane)
    ]
    matches.sort(key=_pool_raw, reverse=True)
    return matches


class _DiscoveryClient(Protocol):
    """The slice of CodePitClient sponsor discovery needs."""

    def list_public_challenges(self) -> Mapping[str, Any]: ...

    def read_eligibility(self, challenge_id: str) -> Mapping[str, Any]: ...


def discover_sponsor_challenge(
    client: _DiscoveryClient,
    *,
    artifact_lane: str,
    allowed_export_target: str | None = None,
) -> str | None:
    """Return the richest *winnable* open sponsor competition on
    ``artifact_lane`` the agent is eligible for, or ``None`` if there is none.

    Candidates are ranked by reward pool. When ``allowed_export_target`` is
    given AND a candidate exposes ``baseline_optimization_methods``, a
    competition whose baseline is already an instance of the allowed export
    class is skipped — the lane cannot strictly beat it, so it would only
    waste compute (the unwinnable-baseline trap, #288/#291). Remaining
    candidates are confirmed top-down against the authoritative
    ``read_eligibility`` endpoint.

    Back-compat: with no ``allowed_export_target``, or for candidates lacking
    the ``baseline_optimization_methods`` signal (older engine), winnability is
    not enforced and behaviour is pure pool-ranking — no regression.
    """
    listing = client.list_public_challenges()
    items = listing.get("items") if isinstance(listing, Mapping) else None
    if not isinstance(items, Sequence):
        return None

    for challenge in rank_sponsor_challenges(items, artifact_lane=artifact_lane):
        if allowed_export_target is not None:
            baseline_methods = challenge.get("baseline_optimization_methods")
            if isinstance(baseline_methods, Sequence) and not isinstance(baseline_methods, str):
                if not _baseline_leaves_headroom(
                    [str(m) for m in baseline_methods],
                    allowed_export_target,
                ):
                    continue
        challenge_id = str(challenge["challenge_id"])
        eligibility = client.read_eligibility(challenge_id)
        if isinstance(eligibility, Mapping) and eligibility.get("eligible"):
            return challenge_id
    return None
