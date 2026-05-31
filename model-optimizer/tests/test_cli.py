import pytest

from codepit_optimizer import cli
from codepit_optimizer.credential_rotation import CredentialRotationResult
from codepit_optimizer.orchestrator import OrchestratorResult
from codepit_optimizer.recipes import RecipeRunResult


def test_main_exits_nonzero_when_no_candidates_succeed(monkeypatch, tmp_path, capsys):
    def fake_run_candidate_recipes(*, source_model, work_dir):
        return [
            RecipeRunResult(
                name="baseline-export",
                output_dir=work_dir / "baseline-export",
                succeeded=False,
                error="export failed",
            ),
            RecipeRunResult(
                name="graph-optimization",
                output_dir=work_dir / "graph-optimization",
                succeeded=False,
                error="optimization failed",
            ),
        ]

    monkeypatch.setattr(cli, "run_candidate_recipes", fake_run_candidate_recipes)
    monkeypatch.setattr(
        "sys.argv",
        ["codepit-model-optimizer", "--work-dir", str(tmp_path)],
    )

    try:
        cli.main()
    except SystemExit as error:
        assert error.code == 1
    else:
        raise AssertionError("main should exit nonzero when every candidate fails")

    output = capsys.readouterr().out
    assert "generated 0 candidate bundle(s)" in output
    assert "failed 2 recipe(s): baseline-export, graph-optimization" in output


def test_generate_recipe_runs_only_selected_recipe(monkeypatch, tmp_path, capsys):
    called = []

    def fake_run_candidate_recipes(*, source_model, work_dir, recipes):
        recipe = list(recipes)[0]
        called.append(recipe.name)
        return [
            RecipeRunResult(
                name=recipe.name,
                output_dir=work_dir / recipe.name,
                succeeded=True,
            )
        ]

    monkeypatch.setattr(cli, "run_candidate_recipes", fake_run_candidate_recipes)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "generate",
            "--work-dir",
            str(tmp_path),
            "--recipe",
            "dynamic-int8",
        ],
    )

    cli.main()

    assert called == ["dynamic-int8"]
    output = capsys.readouterr().out
    assert "dynamic-int8: ok" in output


def test_rotate_credentials_prints_rotation_result(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_rotate(config):
        seen["base_url"] = config.base_url
        seen["agent_id"] = config.agent_id
        seen["session_path"] = config.session_path
        return CredentialRotationResult(
            agent_id="agent_pyopt_1",
            signer_address="0x" + "a" * 40,
            credential_id="cred_new",
            runtime_credential="secret_new",
            superseded_credential_id="cred_old",
            session_path=str(config.session_path),
        )

    monkeypatch.setattr(cli, "rotate_optimizer_credentials", fake_rotate)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "rotate-credentials",
            "--base-url",
            "http://engine.fake",
            "--agent-id",
            "agent_pyopt_1",
            "--private-key",
            "0x" + "1" * 64,
            "--session-path",
            str(tmp_path / "agent.json"),
        ],
    )

    cli.main()

    assert seen["base_url"] == "http://engine.fake"
    assert seen["agent_id"] == "agent_pyopt_1"
    assert seen["session_path"] == tmp_path / "agent.json"
    output = capsys.readouterr().out
    assert '"credential_id": "cred_new"' in output
    assert '"runtime_credential": "secret_new"' in output


def test_run_passes_explicit_client_submission_id(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_run(config):
        seen["client_submission_id"] = config.client_submission_id
        seen["agent_wallet_private_key"] = config.agent_wallet_private_key
        return OrchestratorResult(
            agent_id="agent_pyopt_1",
            signer_address="0x" + "a" * 40,
            challenge_id="challenge_1",
            submission_id="sub_1",
            state="VERIFIED",
            benchmark_target_version="0.1.0",
            chosen_recipe="pre-built",
            bundle_dir=tmp_path,
            client_submission_id=config.client_submission_id,
        )

    monkeypatch.setattr(cli, "run_optimizer_agent", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "run",
            "--base-url",
            "http://engine.fake",
            "--work-dir",
            str(tmp_path),
            "--client-submission-id",
            "retry-key-001",
            "--agent-wallet-private-key",
            "0x" + "2" * 64,
            "--no-session-persist",
        ],
    )

    cli.main()

    assert seen["client_submission_id"] == "retry-key-001"
    assert seen["agent_wallet_private_key"] == "0x" + "2" * 64
    output = capsys.readouterr().out
    assert '"client_submission_id": "retry-key-001"' in output


def test_tiny_chat_run_invokes_run_path_with_parsed_args(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_run(config):
        seen["challenge_id"] = config.challenge_id
        seen["base_model_ref"] = config.base_model_ref
        seen["quantization_profile"] = config.quantization_profile
        seen["session_path"] = config.session_path
        return OrchestratorResult(
            agent_id="agent_pyopt_1",
            signer_address="0x" + "a" * 40,
            challenge_id="ch_gguf_1",
            submission_id="sub_1",
            state="VERIFIED",
            benchmark_target_version="0.1.0",
            chosen_recipe="q4_k_m",
            bundle_dir=tmp_path,
            client_submission_id="pyopt-abc",
        )

    monkeypatch.setattr(cli, "run_tiny_chat_external_agent", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "tiny-chat-run",
            "--base-url",
            "http://engine.fake",
            "--work-dir",
            str(tmp_path),
            "--challenge-id",
            "ch_gguf_1",
            "--quantization-profile",
            "q4_k_m",
            "--no-session-persist",
        ],
    )

    cli.main()

    assert seen["challenge_id"] == "ch_gguf_1"
    assert seen["quantization_profile"] == "q4_k_m"
    assert seen["session_path"] is None
    output = capsys.readouterr().out
    assert '"submission_id": "sub_1"' in output
    assert '"state": "VERIFIED"' in output


def test_tiny_chat_run_target_sponsor_is_passed_to_config(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_run(config):
        seen["target"] = config.target
        seen["challenge_id"] = config.challenge_id
        return OrchestratorResult(
            agent_id="agent_pyopt_1",
            signer_address="0x" + "a" * 40,
            challenge_id="ch_sponsor_1",
            submission_id="sub_1",
            state="VERIFIED",
            benchmark_target_version="0.1.0",
            chosen_recipe="q4_k_m",
            bundle_dir=tmp_path,
            client_submission_id="pyopt-abc",
        )

    monkeypatch.setattr(cli, "run_tiny_chat_external_agent", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "tiny-chat-run",
            "--work-dir",
            str(tmp_path),
            "--target",
            "sponsor",
            "--no-session-persist",
        ],
    )

    cli.main()

    assert seen["target"] == "sponsor"
    # --target sponsor leaves challenge_id unset; discovery resolves it
    assert seen["challenge_id"] is None


def test_tiny_chat_run_allow_unbound_payout_flag_sets_config(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_run(config):
        seen["allow_unbound_payout"] = config.allow_unbound_payout
        return OrchestratorResult(
            agent_id="agent_pyopt_1",
            signer_address="0x" + "a" * 40,
            challenge_id="ch_1",
            submission_id="sub_1",
            state="VERIFIED",
            benchmark_target_version="0.1.0",
            chosen_recipe="q4_k_m",
            bundle_dir=tmp_path,
            client_submission_id="pyopt-abc",
        )

    monkeypatch.setattr(cli, "run_tiny_chat_external_agent", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "tiny-chat-run",
            "--work-dir",
            str(tmp_path),
            "--target",
            "sponsor",
            "--allow-unbound-payout",
            "--no-session-persist",
        ],
    )

    cli.main()
    assert seen["allow_unbound_payout"] is True


def test_tiny_chat_run_defaults_allow_unbound_payout_false(monkeypatch, tmp_path, capsys):
    seen = {}

    def fake_run(config):
        seen["allow_unbound_payout"] = config.allow_unbound_payout
        return OrchestratorResult(
            agent_id="agent_pyopt_1",
            signer_address="0x" + "a" * 40,
            challenge_id="ch_1",
            submission_id="sub_1",
            state="VERIFIED",
            benchmark_target_version="0.1.0",
            chosen_recipe="q4_k_m",
            bundle_dir=tmp_path,
            client_submission_id="pyopt-abc",
        )

    monkeypatch.setattr(cli, "run_tiny_chat_external_agent", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        ["codepit-model-optimizer", "tiny-chat-run", "--work-dir", str(tmp_path), "--no-session-persist"],
    )

    cli.main()
    assert seen["allow_unbound_payout"] is False


def test_tiny_chat_run_passes_prebuilt_gguf_path(monkeypatch, tmp_path, capsys):
    seen = {}
    gguf = tmp_path / "agent-built.gguf"
    gguf.write_bytes(b"GGUF" + b"\x00" * 8)

    def fake_run(config):
        seen["gguf_path"] = config.gguf_path
        return OrchestratorResult(
            agent_id="agent_pyopt_1",
            signer_address="0x" + "a" * 40,
            challenge_id="ch_gguf_1",
            submission_id="sub_1",
            state="VERIFIED",
            benchmark_target_version="0.1.0",
            chosen_recipe="q4_k_m",
            bundle_dir=tmp_path,
            client_submission_id="pyopt-abc",
        )

    monkeypatch.setattr(cli, "run_tiny_chat_external_agent", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "tiny-chat-run",
            "--base-url",
            "http://engine.fake",
            "--work-dir",
            str(tmp_path),
            "--gguf-path",
            str(gguf),
            "--no-session-persist",
        ],
    )

    cli.main()

    assert seen["gguf_path"] == gguf
    assert '"state": "VERIFIED"' in capsys.readouterr().out


def test_modelbook_run_invokes_iteration_with_parsed_args(monkeypatch, capsys):
    seen = {}

    def fake_run_iteration(client, config):
        from codepit_optimizer.modelbook_loop import ModelbookIterationResult
        seen["client_base_url"] = client.base_url
        seen["client_agent_id"] = client.agent_id
        seen["client_credential"] = client.credential
        seen["modelbook_id"] = config.modelbook_id
        seen["recipe_kind"] = config.recipe_kind
        seen["artifact_output_dir"] = config.artifact_output_dir
        seen["submit"] = config.submit
        seen["challenge_id"] = config.challenge_id
        seen["client_submission_id"] = config.client_submission_id
        return ModelbookIterationResult(
            modelbook_id="mb_1",
            training_run_id="run_1",
            recipe_kind="lora",
            decisions_recorded=3,
            events_emitted=6,
            artifact_set_id="art_1",
            challenge_id="ch_tiny",
            submission_id="sub_1",
            submission_state="UPLOADING",
        )

    monkeypatch.setattr(cli, "run_modelbook_iteration", fake_run_iteration)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "modelbook-run",
            "--engine-url",
            "http://engine.test/",
            "--agent-id",
            "agent_1",
            "--credential",
            "bearer-secret",
            "--modelbook-id",
            "mb_1",
            "--recipe-kind",
            "lora",
            "--artifact-output-dir",
            "/tmp/codepit-modelbook-artifacts",
            "--submit",
            "--challenge-id",
            "ch_tiny",
            "--client-submission-id",
            "retry-modelbook-1",
        ],
    )

    cli.main()

    assert seen["client_agent_id"] == "agent_1"
    assert seen["client_credential"] == "bearer-secret"
    assert seen["modelbook_id"] == "mb_1"
    assert seen["recipe_kind"] == "lora"
    assert seen["artifact_output_dir"] == "/tmp/codepit-modelbook-artifacts"
    assert seen["submit"] is True
    assert seen["challenge_id"] == "ch_tiny"
    assert seen["client_submission_id"] == "retry-modelbook-1"

    output = capsys.readouterr().out
    assert '"training_run_id": "run_1"' in output
    assert '"artifact_set_id": "art_1"' in output
    assert '"submission_id": "sub_1"' in output


def test_modelbook_run_exits_nonzero_when_no_modelbook_available(monkeypatch):
    def fake_run_iteration(client, config):
        from codepit_optimizer.modelbook_loop import ModelbookIterationResult
        return ModelbookIterationResult(
            modelbook_id=None,
            training_run_id=None,
            recipe_kind=None,
            skipped_reason="no_available_modelbook",
            stub_training_used=False,
        )

    monkeypatch.setattr(cli, "run_modelbook_iteration", fake_run_iteration)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "modelbook-run",
            "--engine-url",
            "http://engine.test/",
            "--agent-id",
            "agent_1",
            "--credential",
            "bearer-secret",
        ],
    )

    try:
        cli.main()
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("modelbook-run should exit 2 when no Modelbook is available")


def test_modelbook_run_requires_credentials(monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "modelbook-run",
            "--engine-url",
            "http://engine.test/",
        ],
    )
    monkeypatch.delenv("CODEPIT_AGENT_ID", raising=False)
    monkeypatch.delenv("CODEPIT_RUNTIME_CREDENTIAL", raising=False)

    try:
        cli.main()
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("modelbook-run should reject missing credentials")

    err = capsys.readouterr().err
    assert "--agent-id" in err
    assert "--credential" in err


def test_modelbook_run_max_iterations_uses_loop(monkeypatch, capsys):
    captured = {}

    def fake_run_loop(client, config, *, max_iterations, idle_sleep_seconds):
        from codepit_optimizer.modelbook_loop import ModelbookIterationResult
        captured["max_iterations"] = max_iterations
        captured["idle_sleep_seconds"] = idle_sleep_seconds
        return [
            ModelbookIterationResult(
                modelbook_id="mb_1",
                training_run_id=f"run_{i}",
                recipe_kind="lora",
            )
            for i in range(2)
        ]

    monkeypatch.setattr(cli, "run_modelbook_loop", fake_run_loop)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "modelbook-run",
            "--engine-url",
            "http://engine.test/",
            "--agent-id",
            "agent_1",
            "--credential",
            "bearer-secret",
            "--max-iterations",
            "2",
            "--idle-sleep-seconds",
            "0.5",
        ],
    )

    cli.main()

    assert captured == {"max_iterations": 2, "idle_sleep_seconds": 0.5}
    output = capsys.readouterr().out
    assert output.count('"training_run_id"') == 2


def test_claim_agent_subcommand_binds_payout_via_claim_flow(monkeypatch, capsys):
    captured = {}

    def fake_claim_agent_payout(client, **kwargs):
        captured.update(kwargs)
        captured["client_present"] = client is not None
        return {
            "agent_id": kwargs["agent_id"],
            "payout_address": "0x" + "d" * 40,
            "claim_status": "claimed",
        }

    monkeypatch.setattr(cli, "claim_agent_payout", fake_claim_agent_payout)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "claim-agent",
            "--base-url",
            "http://engine.fake",
            "--agent-id",
            "agent-9",
            "--agent-signer-address",
            "0x" + "a" * 40,
            "--claim-token",
            "tok_xyz",
            "--owner-claim-private-key",
            "0x" + "11" * 32,
            "--i-control-the-payout-wallet",
        ],
    )

    cli.main()

    assert captured["agent_id"] == "agent-9"
    assert captured["agent_signer_address"] == "0x" + "a" * 40
    assert captured["claim_token"] == "tok_xyz"
    assert captured["owner_private_key"] == "0x" + "11" * 32
    # the agent's own signer is passed as a forbidden payout address (footgun guard)
    assert "0x" + "a" * 40 in captured["forbidden_payout_addresses"]
    out = capsys.readouterr().out
    # the bound payout address is surfaced so the operator can confirm onboarding
    assert "0x" + "d" * 40 in out
    assert "claimed" in out


def test_claim_agent_subcommand_requires_payout_wallet_acknowledgment(monkeypatch):
    # Without --i-control-the-payout-wallet the kit must refuse to bind a payout
    # wallet, so rewards can't land in a wallet the operator doesn't control / back up.
    called = {"value": False}

    def fake_claim_agent_payout(client, **kwargs):  # pragma: no cover - must not run
        called["value"] = True
        return {"claim_status": "claimed"}

    monkeypatch.setattr(cli, "claim_agent_payout", fake_claim_agent_payout)
    monkeypatch.setattr(
        "sys.argv",
        [
            "codepit-model-optimizer",
            "claim-agent",
            "--base-url",
            "http://engine.fake",
            "--agent-id",
            "agent-9",
            "--agent-signer-address",
            "0x" + "a" * 40,
            "--claim-token",
            "tok_xyz",
            "--owner-claim-private-key",
            "0x" + "11" * 32,
        ],
    )

    with pytest.raises(SystemExit):
        cli.main()
    assert called["value"] is False  # never attempted the bind
