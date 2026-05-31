from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from subprocess import CompletedProcess, run

from .plan import OptimizationExperiment, OptimizationPlan, OptimizationTransform


Runner = Callable[..., CompletedProcess]


@dataclass(frozen=True)
class CandidateRecipe:
    name: str
    methods: list[str]


@dataclass(frozen=True)
class RecipeRunResult:
    name: str
    output_dir: Path
    succeeded: bool
    error: str | None = None


RECIPES = [
    CandidateRecipe("baseline-export", ["baseline-export"]),
    CandidateRecipe("graph-optimization", ["graph-optimization"]),
    CandidateRecipe("dynamic-int8", ["dynamic-int8"]),
]


def candidate_recipe_names() -> list[str]:
    return [recipe.name for recipe in RECIPES]


def get_recipe(name: str) -> CandidateRecipe:
    for recipe in RECIPES:
        if recipe.name == name:
            return recipe
    valid = ", ".join(candidate_recipe_names())
    raise ValueError(f"unknown recipe: {name}. Valid recipes: {valid}")


def build_recipe_command(recipe: CandidateRecipe, source_model: str, output_dir: Path) -> list[str]:
    if recipe.name == "baseline-export":
        return _build_onnx_export_command(source_model, output_dir, optimize=None)
    if recipe.name == "graph-optimization":
        return _build_onnx_export_command(source_model, output_dir, optimize="O2")
    if recipe.name == "dynamic-int8":
        return _build_quantize_command(source_model, output_dir, weight_type="qint8")
    raise ValueError(f"unknown recipe: {recipe.name}")


def run_recipe(
    recipe: CandidateRecipe,
    source_model: str,
    output_dir: Path,
    *,
    runner: Runner = run,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runner(build_recipe_command(recipe, source_model, output_dir), check=True)


def run_candidate_recipes(
    *,
    source_model: str,
    work_dir: Path,
    recipes: Iterable[CandidateRecipe] = RECIPES,
    runner: Runner = run,
) -> list[RecipeRunResult]:
    results: list[RecipeRunResult] = []
    for recipe in recipes:
        output_dir = work_dir / recipe.name
        try:
            run_recipe(recipe, source_model, output_dir, runner=runner)
        except Exception as error:
            results.append(
                RecipeRunResult(
                    name=recipe.name,
                    output_dir=output_dir,
                    succeeded=False,
                    error=str(error),
                )
            )
            continue
        results.append(RecipeRunResult(name=recipe.name, output_dir=output_dir, succeeded=True))
    return results


def run_plan_experiments(
    *,
    plan: OptimizationPlan,
    source_model: str,
    work_dir: Path,
    runner: Runner = run,
) -> list[RecipeRunResult]:
    results: list[RecipeRunResult] = []
    for experiment in plan.experiments:
        output_dir = work_dir / experiment.name
        staging_dir = work_dir / ".plan-staging" / experiment.name
        try:
            run_plan_experiment(
                experiment,
                source_model,
                output_dir,
                staging_dir=staging_dir,
                runner=runner,
            )
        except Exception as error:
            results.append(
                RecipeRunResult(
                    name=experiment.name,
                    output_dir=output_dir,
                    succeeded=False,
                    error=str(error),
                )
            )
            continue
        results.append(RecipeRunResult(name=experiment.name, output_dir=output_dir, succeeded=True))
    return results


def run_plan_experiment(
    experiment: OptimizationExperiment,
    source_model: str,
    output_dir: Path,
    *,
    staging_dir: Path,
    runner: Runner = run,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    commands = build_plan_experiment_commands(
        experiment,
        source_model,
        output_dir,
        staging_dir=staging_dir,
    )
    for command in commands:
        runner(command, check=True)


def build_plan_experiment_commands(
    experiment: OptimizationExperiment,
    source_model: str,
    output_dir: Path,
    *,
    staging_dir: Path,
) -> list[list[str]]:
    executable = list(experiment.executable_transforms)
    commands: list[list[str]] = []
    current_source = source_model

    onnx_export = _first_transform(executable, "onnx_export")
    dynamic_quantization = _first_transform(executable, "dynamic_quantization")

    if onnx_export is not None:
        export_output = output_dir if dynamic_quantization is None else staging_dir / "onnx-export"
        commands.append(
            _build_onnx_export_command(
                current_source,
                export_output,
                optimize=onnx_export.optimize,
            )
        )
        current_source = str(export_output)

    if dynamic_quantization is not None:
        commands.append(
            _build_quantize_command(
                current_source,
                output_dir,
                weight_type=dynamic_quantization.weight_type or "qint8",
            )
        )

    return commands


def summarize_results(results: Sequence[RecipeRunResult]) -> str:
    succeeded = [result.name for result in results if result.succeeded]
    failed = [result.name for result in results if not result.succeeded]
    parts = [f"generated {len(succeeded)} candidate bundle(s)"]
    if failed:
        parts.append(f"failed {len(failed)} recipe(s): {', '.join(failed)}")
    return "; ".join(parts)


def _build_onnx_export_command(
    source_model: str,
    output_dir: Path,
    *,
    optimize: str | None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "codepit_optimizer.export_onnx",
        "--source-model",
        source_model,
    ]
    if optimize:
        command.extend(["--optimize", optimize])
    command.extend(["--output-dir", str(output_dir)])
    return command


def _build_quantize_command(source_model: str, output_dir: Path, *, weight_type: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "codepit_optimizer.quantize",
        "--source-model",
        source_model,
        "--weight-type",
        weight_type,
        "--output-dir",
        str(output_dir),
    ]


def _first_transform(
    transforms: Sequence[OptimizationTransform],
    kind: str,
) -> OptimizationTransform | None:
    for transform in transforms:
        if transform.kind == kind:
            return transform
    return None
