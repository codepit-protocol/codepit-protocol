import sys
from subprocess import CalledProcessError, CompletedProcess

from codepit_optimizer.plan import parse_optimization_plan
from codepit_optimizer.recipes import (
    RECIPES,
    build_plan_experiment_commands,
    build_recipe_command,
    candidate_recipe_names,
    get_recipe,
    run_candidate_recipes,
    run_plan_experiments,
    run_recipe,
    summarize_results,
)


def test_candidate_recipe_names_are_deterministic():
    assert candidate_recipe_names() == [
        "baseline-export",
        "graph-optimization",
        "dynamic-int8",
    ]


def test_get_recipe_returns_known_recipe_or_actionable_error():
    assert get_recipe("graph-optimization").name == "graph-optimization"

    try:
        get_recipe("unknown")
    except ValueError as error:
        assert "unknown recipe: unknown" in str(error)
        assert "baseline-export, graph-optimization, dynamic-int8" in str(error)
    else:
        raise AssertionError("unknown recipe should fail locally")


def test_run_recipe_invokes_expected_export_command(tmp_path):
    calls = []

    def runner(command, check):
        calls.append((command, check))
        return CompletedProcess(command, 0)

    run_recipe(RECIPES[1], "source-model", tmp_path / "candidate", runner=runner)

    command, check = calls[0]
    assert check is True
    assert command[:5] == [
        sys.executable,
        "-m",
        "codepit_optimizer.export_onnx",
        "--source-model",
        "source-model",
    ]
    assert "--optimize" in command
    assert command[-1] == str(tmp_path / "candidate")


def test_dynamic_int8_command_uses_quantize_module(tmp_path):
    command = build_recipe_command(RECIPES[2], "source-model", tmp_path / "candidate")

    assert command[1:3] == ["-m", "codepit_optimizer.quantize"]
    assert "--source-model" in command
    assert "--weight-type" in command
    assert "qint8" in command
    assert str(tmp_path / "candidate") == command[-1]


def test_run_recipe_is_strict_on_candidate_failure(tmp_path):
    def runner(command, check):
        raise CalledProcessError(2, command)

    try:
        run_recipe(RECIPES[0], "source-model", tmp_path / "candidate", runner=runner)
    except CalledProcessError as error:
        assert error.returncode == 2
    else:
        raise AssertionError("run_recipe should surface candidate failures")


def test_run_candidate_recipes_isolates_failures_for_cli(tmp_path):
    calls = []

    def runner(command, check):
        calls.append(command)
        if len(calls) == 2:
            raise CalledProcessError(1, command)
        return CompletedProcess(command, 0)

    results = run_candidate_recipes(source_model="source-model", work_dir=tmp_path, runner=runner)

    assert [result.name for result in results] == candidate_recipe_names()
    assert [result.succeeded for result in results] == [True, False, True]
    assert len(calls) == 3
    assert summarize_results(results) == "generated 2 candidate bundle(s); failed 1 recipe(s): graph-optimization"


def test_build_plan_experiment_commands_stages_export_before_quantization(tmp_path):
    plan = parse_optimization_plan(
        {
            "objective": "minimize_latency_preserve_quality",
            "strategy": "Export with graph opt, then quantize.",
            "max_experiments": 1,
            "experiments": [
                {
                    "name": "o3-quint8",
                    "hypothesis": "O3 graph cleanup then quint8 quantization may improve startup.",
                    "transforms": [
                        {"kind": "onnx_export", "optimize": "O3"},
                        {"kind": "dynamic_quantization", "weight_type": "quint8"},
                    ],
                },
            ],
        }
    )

    commands = build_plan_experiment_commands(
        plan.experiments[0],
        "source-model",
        tmp_path / "o3-quint8",
        staging_dir=tmp_path / ".plan-staging" / "o3-quint8",
    )

    assert len(commands) == 2
    assert commands[0][1:3] == ["-m", "codepit_optimizer.export_onnx"]
    assert "--optimize" in commands[0]
    assert "O3" in commands[0]
    assert commands[0][-1] == str(tmp_path / ".plan-staging" / "o3-quint8" / "onnx-export")
    assert commands[1][1:3] == ["-m", "codepit_optimizer.quantize"]
    assert commands[1][commands[1].index("--source-model") + 1] == commands[0][-1]
    assert commands[1][commands[1].index("--weight-type") + 1] == "quint8"
    assert commands[1][-1] == str(tmp_path / "o3-quint8")


def test_run_plan_experiments_isolates_failed_experiments(tmp_path):
    plan = parse_optimization_plan(
        {
            "objective": "compare_candidates",
            "strategy": "Run one failing candidate and one fallback.",
            "max_experiments": 2,
            "experiments": [
                {
                    "name": "o2-int8",
                    "hypothesis": "candidate may fail during local generation",
                    "transforms": [
                        {"kind": "onnx_export", "optimize": "O2"},
                        {"kind": "dynamic_quantization", "weight_type": "qint8"},
                    ],
                },
                {
                    "name": "graph-only",
                    "hypothesis": "fallback graph candidate",
                    "transforms": [{"kind": "onnx_export", "optimize": "O2"}],
                },
            ],
        }
    )
    calls = []

    def runner(command, check):
        calls.append(command)
        if any(part.endswith(".quantize") for part in command):
            raise CalledProcessError(1, command)
        return CompletedProcess(command, 0)

    results = run_plan_experiments(
        plan=plan,
        source_model="source-model",
        work_dir=tmp_path,
        runner=runner,
    )

    assert [result.name for result in results] == ["o2-int8", "graph-only"]
    assert [result.succeeded for result in results] == [False, True]
    assert "returned non-zero exit status" in (results[0].error or "")
    assert len(calls) == 3
