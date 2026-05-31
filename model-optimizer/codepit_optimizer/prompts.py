"""Prompt builders + JSON schemas for the optimizer Brain.

Kept separate from ``brain.py`` so prompt tuning, schema evolution, and
test snapshots don't have to drag in the full Brain class.

Every prompt MUST:
  - return a single string the Brain passes to the LLM,
  - assume the LLM has no memory across calls (each call carries full context),
  - be deterministic for a given input.

Every JSON schema MUST:
  - validate via ``jsonschema``-style draft-7 semantics,
  - be conservative — extra keys allowed but the required fields must be present,
  - constrain enums where the orchestrator has a closed set of acceptable values.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .plan import OPTIMIZATION_PLAN_SCHEMA

# ---------------------------------------------------------------------------
# pick_recipe
# ---------------------------------------------------------------------------

# The orchestrator currently knows three recipes:
#   - baseline-export   — vanilla ONNX export
#   - graph-optimization — ORT graph optimization (level O2)
#   - dynamic-int8       — per-tensor dynamic int8 quantization
#
# Add new recipes here AND in ``codepit_optimizer.recipes.RECIPES`` together;
# the Brain's choice is constrained to this set.
KNOWN_RECIPE_NAMES: tuple[str, ...] = (
    "baseline-export",
    "graph-optimization",
    "dynamic-int8",
)

#: JSON schema (draft-7-ish) the LLM's response must satisfy.
#:
#: We pin ``recipe_name`` to ``KNOWN_RECIPE_NAMES`` so the model can't invent
#: a recipe the orchestrator doesn't know how to build. ``confidence`` is a
#: float in [0,1]; ``reasoning`` is a short free-text rationale used only for
#: diagnostics.
RECIPE_CHOICE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["recipe_name"],
    "properties": {
        "recipe_name": {
            "type": "string",
            "enum": list(KNOWN_RECIPE_NAMES),
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "reasoning": {"type": "string", "maxLength": 1024},
    },
    "additionalProperties": True,
}


def build_pick_recipe_prompt(
    challenge_spec: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Compose a prompt for ``Brain.pick_recipe``.

    Keep this short and tunable — the prompt is the highest-leverage knob
    in the optimizer. The current shape is intentionally simple so it works
    on cheap-tier models; richer reasoning is a Phase B+ tuning task.
    """
    challenge_blob = json.dumps(_compact(challenge_spec), sort_keys=True, indent=2)
    if history:
        history_blob = json.dumps(
            [_compact(item) for item in history], sort_keys=True, indent=2,
        )
    else:
        history_blob = "[]"

    return (
        "You are CodePit's optimization brain. "
        "Pick the recipe most likely to clear the quality floor while "
        "minimizing latency. Recipes you may pick from:\n"
        f"  {', '.join(KNOWN_RECIPE_NAMES)}.\n\n"
        "Return a JSON object matching this schema:\n"
        f"{json.dumps(RECIPE_CHOICE_SCHEMA, indent=2)}\n\n"
        "Challenge spec:\n"
        f"{challenge_blob}\n\n"
        "Recent history of (recipe, outcome) pairs from this agent:\n"
        f"{history_blob}\n\n"
        "Respond with the JSON object only — no prose."
    )


def build_optimization_plan_prompt(
    challenge_spec: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Compose a prompt for ``Brain.plan_optimization``.

    This prompt deliberately gives the LLM strategic freedom: it can choose
    experiment order, transform combinations, and risk posture. The trusted
    executor still ignores free-text reasoning and only runs validated
    transform fields.
    """
    challenge_blob = json.dumps(_compact(challenge_spec), sort_keys=True, indent=2)
    if history:
        history_blob = json.dumps(
            [_compact(item) for item in history], sort_keys=True, indent=2,
        )
    else:
        history_blob = "[]"

    return (
        "You are CodePit's model optimization brain. Think like an optimizer, "
        "not a fixed recipe selector. Your job is to propose 1-3 experiments "
        "that can improve a small open-weight model for the benchmark target.\n\n"
        "You may be creative about hypotheses, experiment order, and tradeoffs. "
        "Use the free-text fields to explain your reasoning. The executor will "
        "only run the validated transform fields, and the official verifier is "
        "the only source of truth for public claims.\n\n"
        "Allowed executable transforms:\n"
        "- onnx_export with optimize null, O1, O2, O3, or O4\n"
        "- dynamic_quantization with weight_type qint8 or quint8 and per_channel false\n"
        "Allowed non-executable transforms:\n"
        "- package_layout for packaging intent; include_tokenizer_files must be true\n"
        "- metadata for notes only\n\n"
        "Return a JSON object matching this schema:\n"
        f"{json.dumps(OPTIMIZATION_PLAN_SCHEMA, indent=2)}\n\n"
        "Challenge spec:\n"
        f"{challenge_blob}\n\n"
        "Recent compact history of prior experiments and outcomes:\n"
        f"{history_blob}\n\n"
        "Respond with the JSON object only — no prose."
    )


def _compact(item: Mapping[str, Any]) -> dict[str, Any]:
    """Drop noisy fields before stuffing into the prompt.

    The challenge spec carries fields the LLM doesn't need to read (audit
    timestamps, internal ids). Strip them so we don't burn tokens.
    """
    out: dict[str, Any] = {}
    skip = {"created_at", "updated_at", "engine_internal_id"}
    for key, value in item.items():
        if key in skip:
            continue
        out[key] = value
    return out
