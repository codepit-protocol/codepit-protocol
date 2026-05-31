"""Modal app definition for the CodePit V2 optimizer agent.

This file is the production target for `engine/src/v2/runtime/modal-worker-pool.ts`.
The TypeScript control plane spawns this function with the agent's signer
private key + runtime credential + engine base URL, and the function then
hands off to `codepit_optimizer.orchestrator.run_optimizer_agent_forever`.

Deploy with:

    cd agents/model-optimizer
    modal deploy modal_app.py

The deployed app/function names must stay in sync with `ModalWorkerPoolConfig`
on the engine side (see `modal-worker-pool.ts` defaults).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import modal


# Build the image from the in-tree Dockerfile so the kit + extras are baked
# in. The build context is the `agents/model-optimizer/` directory.
image = modal.Image.from_dockerfile(
    "Dockerfile",
    context_mount=modal.Mount.from_local_dir(
        str(Path(__file__).parent),
        remote_path="/build-ctx",
    ),
)

app = modal.App("codepit-optimizer-agent")


@app.function(
    image=image,
    # Up to 1 hour per iteration is plenty for an open-weight optimizer pass;
    # `run-forever` will exit cleanly on SIGTERM if Modal scales the container
    # down. The control plane reschedules via a fresh `provisionWorker` call.
    timeout=60 * 60,
    # The agent is a network client only; CPU image is sufficient. If model
    # export needs a GPU later, pass `gpu="t4"` (or higher).
    cpu=2.0,
    memory=4096,
    # Scale-to-zero: idle containers are cheap. `min_containers=0` is the
    # libmodal default; we set it explicitly so future tuning is obvious.
    min_containers=0,
)
def run_agent(
    agent_id: str,
    agent_signer_pk: str,
    runtime_credential: str,
    base_url: str,
    brain_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the supervised optimizer agent until killed.

    Inputs are forwarded verbatim from `ModalWorkerPool.provisionWorker`.
    The engine has already provisioned the managed agent and bearer
    credential. The worker reuses that session directly and never
    registers a second external agent.

    Returns the run-forever summary. Modal records this on the function
    call so the control plane can fetch it for postmortem.
    """

    # Lazy import so module-load time is dominated by Modal, not by the
    # kit's heavy ML deps.
    from codepit_optimizer.orchestrator import (
        ForeverConfig,
        OrchestratorConfig,
        default_lane_runners,
        run_optimizer_agent_forever,
    )

    work_root = Path(tempfile.mkdtemp(prefix=f"agent-{agent_id}-"))

    # Brain config: the kit currently uses upstream HF/Transformers and
    # doesn't read from a brain provider directly. Once the optimizer kit
    # adds an LLM-driven recipe selector, this is the env handoff point.
    if brain_config is not None:
        provider = brain_config.get("provider")
        api_key = brain_config.get("apiKey") or brain_config.get("api_key")
        proxy_url = brain_config.get("proxyUrl") or brain_config.get("proxy_url")
        if provider:
            os.environ["CODEPIT_BRAIN_PROVIDER"] = str(provider)
        if api_key:
            os.environ["CODEPIT_BRAIN_API_KEY"] = str(api_key)
        if proxy_url:
            os.environ["CODEPIT_BRAIN_PROXY_URL"] = str(proxy_url)
        if "model" in brain_config and brain_config["model"]:
            os.environ["CODEPIT_BRAIN_MODEL"] = str(brain_config["model"])

    base_config = OrchestratorConfig(
        base_url=base_url,
        work_dir=work_root,
        private_key=agent_signer_pk,
        agent_id=agent_id,
        runtime_credential=runtime_credential,
        challenge_id=None,
        # Persist under the work dir so each Modal container starts clean.
        session_path=work_root / "session.json",
    )
    os.environ.setdefault("CODEPIT_V2_AGENT_PRIVATE_KEY", agent_signer_pk)
    # Hand the bearer to the kit's session bootstrap path.
    os.environ.setdefault("CODEPIT_V2_RUNTIME_CREDENTIAL", runtime_credential)

    # Production managed-agent path opts into the lane-runner registry so a
    # single spawned worker can serve any artifact lane the engine surfaces
    # — no per-deployment recipe lock-in (#87).
    forever = ForeverConfig(
        base_config=base_config,
        lane_runners=default_lane_runners(),
    )
    summary = run_optimizer_agent_forever(forever)

    return {
        "agent_id": agent_id,
        "iterations_started": summary.iterations_started,
        "iterations_completed": summary.iterations_completed,
        "transient_error_count": summary.transient_error_count,
        "terminal_state_counts": summary.terminal_state_counts,
        "stopped_reason": summary.stopped_reason,
        "last_error": summary.last_error,
    }
