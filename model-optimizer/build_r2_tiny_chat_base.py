#!/usr/bin/env python3
"""Build the tiny-chat FP16 *base* GGUF and publish it to Cloudflare R2 (slice A / #271).

Downloads the base model at a pinned revision, converts it to an FP16 GGUF with a
pinned llama.cpp `convert_hf_to_gguf.py`, records provenance (sha256,
base_model_ref, source revision, llama.cpp version), and (with --upload) uploads
to R2 under the stable key, verifying the uploaded object's sha256 round-trips.

The heavy ML/toolchain legs are only touched in the build path, so `--help` and
the provenance helpers work without them. Run where the pinned llama.cpp convert
toolchain + R2 credentials exist (managed worker / CI / one-off HF Job), not a
version-fragile laptop.

Prereqs:
  pip install -e .[optimize]      # huggingface-hub for the model download
  a pinned llama.cpp checkout     # for convert_hf_to_gguf.py
  pip install boto3               # for the R2 upload (--upload)
  env: R2_ENDPOINT / R2_ACCESS_KEY_ID / R2_SECRET_ACCESS_KEY / R2_BUCKET
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from codepit_optimizer.base_gguf_build import (
    R2ObjectStore,
    build_base_gguf_fp16,
    default_base_gguf_key,
    publish_base_gguf,
)


def _download_base_model(base_model_ref: str, revision: str, dest: Path) -> Path:
    """Download the HF base model checkout used as convert input (lazy import)."""
    from huggingface_hub import snapshot_download  # lazy

    repo_id = base_model_ref.replace("hf://", "")
    local_dir = snapshot_download(repo_id=repo_id, revision=revision, local_dir=str(dest))
    return Path(local_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="hf://Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--source-revision", default="main")
    parser.add_argument(
        "--convert-script",
        default=os.environ.get("CODEPIT_GGUF_CONVERT_SCRIPT"),
        help="Path to the pinned llama.cpp convert_hf_to_gguf.py (or env CODEPIT_GGUF_CONVERT_SCRIPT)",
    )
    parser.add_argument(
        "--llama-cpp-version",
        default=os.environ.get("CODEPIT_LLAMA_CPP_VERSION", "unknown"),
        help="Pinned llama.cpp version recorded in provenance (e.g. b9270)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("./tiny-chat-base-build"))
    parser.add_argument("--key", default=default_base_gguf_key())
    parser.add_argument("--bucket", default=os.environ.get("R2_BUCKET"))
    parser.add_argument("--endpoint", default=os.environ.get("R2_ENDPOINT"))
    parser.add_argument("--upload", action="store_true", help="upload to R2 after build")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Use an already-downloaded base model dir instead of fetching from HF",
    )
    args = parser.parse_args()

    if not args.convert_script:
        print("error: --convert-script (or CODEPIT_GGUF_CONVERT_SCRIPT) required", file=sys.stderr)
        raise SystemExit(2)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.model_dir or _download_base_model(
        args.base_model, args.source_revision, args.out_dir / "model"
    )

    result = build_base_gguf_fp16(
        base_model_dir=model_dir,
        out_gguf=args.out_dir / "fp16.gguf",
        base_model_ref=args.base_model,
        source_revision=args.source_revision,
        convert_script=args.convert_script,
        llama_cpp_version=args.llama_cpp_version,
    )

    prov_path = args.out_dir / "provenance.json"
    prov_path.write_text(json.dumps(result.provenance, indent=2, sort_keys=True))
    print(json.dumps(result.provenance, indent=2, sort_keys=True))

    if args.upload:
        if not (args.bucket and args.endpoint):
            print("error: --bucket and --endpoint (or R2_BUCKET/R2_ENDPOINT) required for upload", file=sys.stderr)
            raise SystemExit(2)
        store = R2ObjectStore(
            endpoint=args.endpoint,
            access_key_id=os.environ.get("R2_ACCESS_KEY_ID", ""),
            secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY", ""),
        )
        summary = publish_base_gguf(result, object_store=store, bucket=args.bucket, key=args.key)
        print(json.dumps({"uploaded": summary}, indent=2, sort_keys=True))

    print("done")


if __name__ == "__main__":
    main()
