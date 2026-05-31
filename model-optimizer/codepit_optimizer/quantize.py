from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic

from .export_onnx import export_model_to_onnx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weight-type", choices=["qint8", "quint8"], default="qint8")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="codepit-onnx-") as tmp:
        source_onnx = _resolve_source_onnx(args.source_model, Path(tmp))
        quantize_dynamic(
            model_input=source_onnx,
            model_output=output_dir / "model.onnx",
            weight_type=_quant_type(args.weight_type),
        )
        _copy_supporting_files(source_onnx.parent, output_dir)


def _quant_type(weight_type: str) -> QuantType:
    if weight_type == "qint8":
        return QuantType.QInt8
    if weight_type == "quint8":
        return QuantType.QUInt8
    raise ValueError(f"unsupported weight_type: {weight_type}")


def _resolve_source_onnx(source_model: str, tmp_dir: Path) -> Path:
    source = Path(source_model)
    if source.is_file():
        return source
    if source.is_dir():
        candidate = source / "model.onnx"
        if not candidate.exists():
            raise FileNotFoundError(f"missing model.onnx in {source}")
        return candidate

    export_dir = tmp_dir / "export"
    export_model_to_onnx(source_model=source_model, output_dir=export_dir)
    return export_dir / "model.onnx"


def _copy_supporting_files(source_dir: Path, output_dir: Path) -> None:
    for name in [
        "model.onnx_data",
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.txt",
        "vocab.json",
        "merges.txt",
    ]:
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, output_dir / name)


if __name__ == "__main__":
    main()
