from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from subprocess import run


def run_preflight(
    engine_dir: Path,
    bundle_dir: Path,
    chromium_path: str | None = None,
    *,
    extra_env: Mapping[str, str] | None = None,
) -> dict:
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    if chromium_path:
        env["CODEPIT_CHROMIUM_EXECUTABLE_PATH"] = chromium_path

    proc = run(
        ["bun", "run", "scripts/smoke-v2-verifier-reference.ts", str(bundle_dir)],
        cwd=engine_dir,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)
