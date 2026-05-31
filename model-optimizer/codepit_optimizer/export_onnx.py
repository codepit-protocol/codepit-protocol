from __future__ import annotations

import argparse
from pathlib import Path


def export_model_to_onnx(
    *,
    source_model: str,
    output_dir: Path,
    optimize: str | None = None,
) -> None:
    # Import lazily so lightweight unit-test installs do not need the ML stack.
    from optimum.exporters.onnx import main_export

    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "model_name_or_path": source_model,
        "output": output_dir,
        "task": "feature-extraction",
    }
    if optimize:
        kwargs["optimize"] = optimize
    main_export(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--optimize", choices=["O1", "O2", "O3", "O4"])
    args = parser.parse_args()

    export_model_to_onnx(
        source_model=args.source_model,
        output_dir=Path(args.output_dir),
        optimize=args.optimize,
    )


if __name__ == "__main__":
    main()
