from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


MAX_EXPERIMENTS = 3
ALLOWED_TRANSFORM_KINDS = frozenset(
    {"onnx_export", "dynamic_quantization", "package_layout", "metadata"},
)
ALLOWED_ONNX_OPTIMIZE_LEVELS = frozenset({None, "O1", "O2", "O3", "O4"})
ALLOWED_QUANT_WEIGHT_TYPES = frozenset({"qint8", "quint8"})
_SAFE_EXPERIMENT_NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


class OptimizationPlanError(ValueError):
    """Raised when an LLM optimization plan is not safe to execute."""


@dataclass(frozen=True)
class OptimizationTransform:
    kind: str
    optimize: str | None = None
    weight_type: str | None = None
    per_channel: bool = False
    external_data: bool | None = None
    include_tokenizer_files: bool | None = None
    notes: str = ""

    @property
    def executable(self) -> bool:
        return self.kind in {"onnx_export", "dynamic_quantization"}


@dataclass(frozen=True)
class OptimizationExperiment:
    name: str
    hypothesis: str
    transforms: tuple[OptimizationTransform, ...]
    expected_tradeoff: str = ""
    risk_notes: str = ""
    why_this_next: str = ""

    @property
    def executable_transforms(self) -> tuple[OptimizationTransform, ...]:
        return tuple(transform for transform in self.transforms if transform.executable)


@dataclass(frozen=True)
class OptimizationPlan:
    objective: str
    strategy: str
    max_experiments: int
    experiments: tuple[OptimizationExperiment, ...]
    fallback_strategy: str = ""
    stop_rules: Mapping[str, Any] = field(default_factory=dict)
    legacy_recipe_name: str | None = None


OPTIMIZATION_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["objective", "strategy", "max_experiments", "experiments"],
    "properties": {
        "objective": {"type": "string", "minLength": 1, "maxLength": 160},
        "strategy": {"type": "string", "minLength": 1, "maxLength": 2048},
        "max_experiments": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_EXPERIMENTS,
        },
        "experiments": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_EXPERIMENTS,
            "items": {
                "type": "object",
                "required": ["name", "hypothesis", "transforms"],
                "properties": {
                    "name": {
                        "type": "string",
                        "pattern": _SAFE_EXPERIMENT_NAME.pattern,
                    },
                    "hypothesis": {"type": "string", "minLength": 1, "maxLength": 1024},
                    "transforms": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "oneOf": [
                                {
                                    "type": "object",
                                    "required": ["kind"],
                                    "properties": {
                                        "kind": {"const": "onnx_export"},
                                        "optimize": {
                                            "type": ["string", "null"],
                                            "enum": [None, "O1", "O2", "O3", "O4"],
                                        },
                                    },
                                    "additionalProperties": False,
                                },
                                {
                                    "type": "object",
                                    "required": ["kind"],
                                    "properties": {
                                        "kind": {"const": "dynamic_quantization"},
                                        "weight_type": {
                                            "type": "string",
                                            "enum": ["qint8", "quint8"],
                                        },
                                        "per_channel": {"type": "boolean", "const": False},
                                    },
                                    "additionalProperties": False,
                                },
                                {
                                    "type": "object",
                                    "required": ["kind"],
                                    "properties": {
                                        "kind": {"const": "package_layout"},
                                        "external_data": {"type": "boolean"},
                                        "include_tokenizer_files": {
                                            "type": "boolean",
                                            "const": True,
                                        },
                                    },
                                    "additionalProperties": False,
                                },
                                {
                                    "type": "object",
                                    "required": ["kind"],
                                    "properties": {
                                        "kind": {"const": "metadata"},
                                        "notes": {"type": "string", "maxLength": 2048},
                                    },
                                    "additionalProperties": False,
                                },
                            ],
                        },
                    },
                    "expected_tradeoff": {"type": "string", "maxLength": 1024},
                    "risk_notes": {"type": "string", "maxLength": 1024},
                    "why_this_next": {"type": "string", "maxLength": 1024},
                },
                "additionalProperties": False,
            },
        },
        "fallback_strategy": {"type": "string", "maxLength": 1024},
        "stop_rules": {"type": "object"},
    },
    "additionalProperties": False,
}


def parse_optimization_plan(raw: str | Mapping[str, Any]) -> OptimizationPlan:
    payload = _decode_payload(raw)
    if "recipe_name" in payload:
        recipe_name = _required_str(payload, "recipe_name")
        reasoning = str(payload.get("reasoning") or "legacy recipe choice")
        return plan_from_recipe_name(recipe_name, reasoning=reasoning)

    allowed_top_level = {
        "objective",
        "strategy",
        "max_experiments",
        "experiments",
        "fallback_strategy",
        "stop_rules",
    }
    _reject_unknown_keys(payload, allowed_top_level, "plan")

    objective = _required_str(payload, "objective")
    strategy = _required_str(payload, "strategy")
    max_experiments = _required_int(payload, "max_experiments")
    if not 1 <= max_experiments <= MAX_EXPERIMENTS:
        raise OptimizationPlanError(
            f"max_experiments must be between 1 and {MAX_EXPERIMENTS}",
        )

    experiments_payload = payload.get("experiments")
    if not isinstance(experiments_payload, Sequence) or isinstance(experiments_payload, (str, bytes)):
        raise OptimizationPlanError("experiments must be a non-empty array")
    if not 1 <= len(experiments_payload) <= max_experiments:
        raise OptimizationPlanError(
            "experiments length must be between 1 and max_experiments",
        )

    experiments = tuple(
        _parse_experiment(item, index)
        for index, item in enumerate(experiments_payload, start=1)
    )
    names = [experiment.name for experiment in experiments]
    if len(names) != len(set(names)):
        raise OptimizationPlanError("experiment names must be unique")

    stop_rules = payload.get("stop_rules") or {}
    if not isinstance(stop_rules, Mapping):
        raise OptimizationPlanError("stop_rules must be an object")

    return OptimizationPlan(
        objective=objective,
        strategy=strategy,
        max_experiments=max_experiments,
        experiments=experiments,
        fallback_strategy=str(payload.get("fallback_strategy") or ""),
        stop_rules=dict(stop_rules),
    )


def plan_from_recipe_name(recipe_name: str, *, reasoning: str = "") -> OptimizationPlan:
    if recipe_name == "baseline-export":
        experiment = OptimizationExperiment(
            name="baseline-export",
            hypothesis=reasoning or "Export the baseline ONNX model unchanged.",
            transforms=(OptimizationTransform(kind="onnx_export", optimize=None),),
            expected_tradeoff="lowest quality risk, limited speed improvement",
        )
    elif recipe_name == "graph-optimization":
        experiment = OptimizationExperiment(
            name="graph-optimization",
            hypothesis=reasoning or "Apply graph optimization while preserving embedding behavior.",
            transforms=(OptimizationTransform(kind="onnx_export", optimize="O2"),),
            expected_tradeoff="lower latency with low quality risk",
        )
    elif recipe_name == "dynamic-int8":
        experiment = OptimizationExperiment(
            name="dynamic-int8",
            hypothesis=reasoning or "Use dynamic int8 quantization to reduce size and improve latency.",
            transforms=(
                OptimizationTransform(
                    kind="dynamic_quantization",
                    weight_type="qint8",
                    per_channel=False,
                ),
            ),
            expected_tradeoff="smaller and likely faster, possible quality drop",
        )
    else:
        raise OptimizationPlanError(f"unknown recipe_name {recipe_name!r}")

    return OptimizationPlan(
        objective="legacy_recipe_compatibility",
        strategy=f"Run legacy recipe {recipe_name}.",
        max_experiments=1,
        experiments=(experiment,),
        fallback_strategy="Use baseline-export if the legacy recipe cannot be executed.",
        stop_rules={"submit_first_verified_improvement": False},
        legacy_recipe_name=recipe_name,
    )


def safe_default_plan(*, reason: str = "") -> OptimizationPlan:
    return plan_from_recipe_name(
        "baseline-export",
        reasoning=reason or "Safe fallback after invalid or unavailable brain output.",
    )


def _decode_payload(raw: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as error:
            raise OptimizationPlanError(f"brain returned non-JSON output: {error}") from error
    elif isinstance(raw, Mapping):
        decoded = dict(raw)
    else:
        raise OptimizationPlanError(
            f"brain output was not a JSON object: {type(raw).__name__}",
        )
    if not isinstance(decoded, Mapping):
        raise OptimizationPlanError(
            f"brain output was not a JSON object: {type(decoded).__name__}",
        )
    return dict(decoded)


def _parse_experiment(raw: object, index: int) -> OptimizationExperiment:
    if not isinstance(raw, Mapping):
        raise OptimizationPlanError(f"experiment {index} must be an object")
    payload = dict(raw)
    _reject_unknown_keys(
        payload,
        {
            "name",
            "hypothesis",
            "transforms",
            "expected_tradeoff",
            "risk_notes",
            "why_this_next",
        },
        f"experiment {index}",
    )

    name = _required_str(payload, "name")
    if not _SAFE_EXPERIMENT_NAME.match(name):
        raise OptimizationPlanError(
            f"experiment name {name!r} must match {_SAFE_EXPERIMENT_NAME.pattern}",
        )
    transforms_payload = payload.get("transforms")
    if not isinstance(transforms_payload, Sequence) or isinstance(transforms_payload, (str, bytes)):
        raise OptimizationPlanError(f"experiment {name} transforms must be an array")
    if not transforms_payload:
        raise OptimizationPlanError(f"experiment {name} must include at least one transform")

    transforms = tuple(_parse_transform(item, name) for item in transforms_payload)
    executable = [transform.kind for transform in transforms if transform.executable]
    if not executable:
        raise OptimizationPlanError(
            f"experiment {name} must include onnx_export or dynamic_quantization",
        )
    if executable not in (
        ["onnx_export"],
        ["dynamic_quantization"],
        ["onnx_export", "dynamic_quantization"],
    ):
        raise OptimizationPlanError(
            f"experiment {name} executable transforms must be onnx_export, "
            "dynamic_quantization, or onnx_export followed by dynamic_quantization",
        )

    return OptimizationExperiment(
        name=name,
        hypothesis=_required_str(payload, "hypothesis"),
        transforms=transforms,
        expected_tradeoff=str(payload.get("expected_tradeoff") or ""),
        risk_notes=str(payload.get("risk_notes") or ""),
        why_this_next=str(payload.get("why_this_next") or ""),
    )


def _parse_transform(raw: object, experiment_name: str) -> OptimizationTransform:
    if not isinstance(raw, Mapping):
        raise OptimizationPlanError(f"transform in {experiment_name} must be an object")
    payload = dict(raw)
    kind = _required_str(payload, "kind")
    if kind not in ALLOWED_TRANSFORM_KINDS:
        raise OptimizationPlanError(f"unknown transform kind {kind!r}")

    if kind == "onnx_export":
        _reject_unknown_keys(payload, {"kind", "optimize"}, "onnx_export")
        optimize = payload.get("optimize")
        if optimize not in ALLOWED_ONNX_OPTIMIZE_LEVELS:
            raise OptimizationPlanError(f"unsupported onnx_export optimize level {optimize!r}")
        return OptimizationTransform(kind=kind, optimize=optimize)

    if kind == "dynamic_quantization":
        _reject_unknown_keys(payload, {"kind", "weight_type", "per_channel"}, "dynamic_quantization")
        weight_type = str(payload.get("weight_type") or "qint8")
        if weight_type not in ALLOWED_QUANT_WEIGHT_TYPES:
            raise OptimizationPlanError(f"unsupported dynamic_quantization weight_type {weight_type!r}")
        per_channel = bool(payload.get("per_channel", False))
        if per_channel:
            raise OptimizationPlanError("dynamic_quantization per_channel=true is not enabled in this slice")
        return OptimizationTransform(
            kind=kind,
            weight_type=weight_type,
            per_channel=False,
        )

    if kind == "package_layout":
        _reject_unknown_keys(
            payload,
            {"kind", "external_data", "include_tokenizer_files"},
            "package_layout",
        )
        include_tokenizer_files = payload.get("include_tokenizer_files", True)
        if include_tokenizer_files is not True:
            raise OptimizationPlanError("package_layout.include_tokenizer_files must be true")
        external_data = payload.get("external_data")
        if external_data is not None and not isinstance(external_data, bool):
            raise OptimizationPlanError("package_layout.external_data must be a boolean")
        return OptimizationTransform(
            kind=kind,
            external_data=external_data,
            include_tokenizer_files=True,
        )

    _reject_unknown_keys(payload, {"kind", "notes"}, "metadata")
    return OptimizationTransform(kind=kind, notes=str(payload.get("notes") or ""))


def _required_str(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise OptimizationPlanError(f"{key} must be a non-empty string")
    return value.strip()


def _required_int(payload: Mapping[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise OptimizationPlanError(f"{key} must be an integer")
    return value


def _reject_unknown_keys(payload: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise OptimizationPlanError(f"{context} contains unsupported key(s): {', '.join(unknown)}")


__all__ = [
    "ALLOWED_ONNX_OPTIMIZE_LEVELS",
    "ALLOWED_QUANT_WEIGHT_TYPES",
    "ALLOWED_TRANSFORM_KINDS",
    "MAX_EXPERIMENTS",
    "OPTIMIZATION_PLAN_SCHEMA",
    "OptimizationExperiment",
    "OptimizationPlan",
    "OptimizationPlanError",
    "OptimizationTransform",
    "parse_optimization_plan",
    "plan_from_recipe_name",
    "safe_default_plan",
]
