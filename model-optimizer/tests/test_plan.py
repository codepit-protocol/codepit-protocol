import json

import pytest

from codepit_optimizer.plan import (
    MAX_EXPERIMENTS,
    OPTIMIZATION_PLAN_SCHEMA,
    OptimizationPlanError,
    parse_optimization_plan,
    plan_from_recipe_name,
    safe_default_plan,
)


def _valid_plan() -> dict:
    return {
        "objective": "minimize_latency_preserve_quality",
        "strategy": "Try quantization first, then graph-only if quality risk is too high.",
        "max_experiments": 2,
        "experiments": [
            {
                "name": "o2-int8",
                "hypothesis": "O2 plus int8 may reduce latency while keeping quality above floor.",
                "transforms": [
                    {"kind": "onnx_export", "optimize": "O2"},
                    {"kind": "dynamic_quantization", "weight_type": "qint8", "per_channel": False},
                    {"kind": "package_layout", "external_data": False, "include_tokenizer_files": True},
                    {"kind": "metadata", "notes": "higher risk speed candidate"},
                ],
                "expected_tradeoff": "faster and smaller, possible quality loss",
                "risk_notes": "fall back to graph-only if quality drops",
                "why_this_next": "highest likely speed win",
            },
            {
                "name": "o3-graph-only",
                "hypothesis": "Graph-only optimization has lower embedding quality risk.",
                "transforms": [{"kind": "onnx_export", "optimize": "O3"}],
                "expected_tradeoff": "smaller latency win but lower risk",
            },
        ],
        "fallback_strategy": "Submit graph-only if quantized candidate fails local load.",
        "stop_rules": {
            "submit_first_verified_improvement": False,
            "stop_if_quality_below_floor": True,
        },
    }


def test_parse_optimization_plan_accepts_multi_experiment_reasoning() -> None:
    plan = parse_optimization_plan(json.dumps(_valid_plan()))

    assert plan.objective == "minimize_latency_preserve_quality"
    assert plan.max_experiments == 2
    assert len(plan.experiments) == 2
    assert plan.experiments[0].name == "o2-int8"
    assert [t.kind for t in plan.experiments[0].transforms] == [
        "onnx_export",
        "dynamic_quantization",
        "package_layout",
        "metadata",
    ]
    assert plan.experiments[0].executable_transforms[1].weight_type == "qint8"
    assert plan.stop_rules["stop_if_quality_below_floor"] is True


def test_plan_schema_caps_experiment_budget() -> None:
    assert OPTIMIZATION_PLAN_SCHEMA["properties"]["max_experiments"]["maximum"] == MAX_EXPERIMENTS


@pytest.mark.parametrize(
    "mutate, match",
    [
        (
            lambda payload: payload.update({"shell": "rm -rf /"}),
            "unsupported key",
        ),
        (
            lambda payload: payload["experiments"][0]["transforms"].append(
                {"kind": "shell", "command": "python anything.py"},
            ),
            "unknown transform",
        ),
        (
            lambda payload: payload["experiments"][0]["transforms"][1].update(
                {"per_channel": True},
            ),
            "per_channel",
        ),
        (
            lambda payload: payload["experiments"][0]["transforms"][0].update(
                {"optimize": "O9"},
            ),
            "optimize",
        ),
        (
            lambda payload: payload["experiments"][0].update({"name": "../escape"}),
            "experiment name",
        ),
        (
            lambda payload: payload["experiments"][0].update(
                {"transforms": [{"kind": "metadata", "notes": "no executable work"}]},
            ),
            "must include onnx_export or dynamic_quantization",
        ),
        (
            lambda payload: payload["experiments"][0].update(
                {
                    "transforms": [
                        {"kind": "dynamic_quantization", "weight_type": "qint8"},
                        {"kind": "onnx_export", "optimize": "O2"},
                    ],
                },
            ),
            "executable transforms",
        ),
        (
            lambda payload: payload["experiments"][0]["transforms"][2].update(
                {"include_tokenizer_files": False},
            ),
            "include_tokenizer_files",
        ),
    ],
)
def test_parse_optimization_plan_rejects_unsafe_shapes(mutate, match: str) -> None:
    payload = _valid_plan()
    mutate(payload)

    with pytest.raises(OptimizationPlanError, match=match):
        parse_optimization_plan(payload)


def test_legacy_recipe_choice_maps_to_plan() -> None:
    plan = parse_optimization_plan(
        {"recipe_name": "graph-optimization", "reasoning": "low risk graph cleanup"},
    )

    assert plan.legacy_recipe_name == "graph-optimization"
    assert plan.experiments[0].name == "graph-optimization"
    assert plan.experiments[0].transforms[0].kind == "onnx_export"
    assert plan.experiments[0].transforms[0].optimize == "O2"


def test_safe_default_plan_is_baseline_export() -> None:
    plan = safe_default_plan(reason="provider down")

    assert plan.legacy_recipe_name == "baseline-export"
    assert plan.experiments[0].name == "baseline-export"
    assert "provider down" in plan.experiments[0].hypothesis


def test_unknown_legacy_recipe_fails_closed() -> None:
    with pytest.raises(OptimizationPlanError, match="unknown recipe_name"):
        plan_from_recipe_name("magic")
