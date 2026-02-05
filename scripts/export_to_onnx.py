"""Export a Hugging Face audio classifier to ONNX."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from numbers import Number

import torch


def _quantize_model(model_path: Path, mode: str) -> None:
    if mode == "none":
        return

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError("onnxruntime>=1.16 is required for model quantization") from exc

    tmp_path = model_path.with_name(f"{model_path.stem}.tmp{model_path.suffix}")
    quantize_dynamic(
        model_input=str(model_path),
        model_output=str(tmp_path),
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
    )
    os.replace(tmp_path, model_path)


def _ensure_rms_norm() -> None:
    if hasattr(torch, "rms_norm"):
        return

    def _rms_norm(
        input_tensor: torch.Tensor,
        normalized_shape: int | tuple[int, ...],
        weight: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        if isinstance(normalized_shape, Number):
            normalized_shape = (int(normalized_shape),)

        dims = tuple(range(-len(normalized_shape), 0))
        variance = input_tensor.pow(2).mean(dim=dims, keepdim=True)
        output = input_tensor * torch.rsqrt(variance + eps)

        def _reshape(param: torch.Tensor | None) -> torch.Tensor | None:
            if param is None:
                return None
            expand_shape = (1,) * (output.dim() - len(normalized_shape)) + normalized_shape
            return param.view(expand_shape)

        weight_view = _reshape(weight)
        bias_view = _reshape(bias)

        if weight_view is not None:
            output = output * weight_view
        if bias_view is not None:
            output = output + bias_view
        return output

    torch.rms_norm = _rms_norm  # type: ignore[attr-defined]


_ensure_rms_norm()


def _disable_torch_dynamo() -> None:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    try:
        import torch._dynamo  # type: ignore

        torch._dynamo.config.suppress_errors = True
    except Exception:
        pass


def _patch_fsdp_check() -> None:
    try:
        from transformers.integrations import fsdp as transformers_fsdp  # type: ignore

        transformers_fsdp.is_fsdp_managed_module = lambda module: False  # type: ignore
    except Exception:
        pass

_disable_torch_dynamo()
_patch_fsdp_check()

from optimum.exporters.onnx import main_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a Transformers audio model to ONNX format.")
    parser.add_argument(
        "--model-name-or-path",
        default="MelodyMachine/Deepfake-audio-detection-V2",
        help="Hugging Face model ID or local path to a fine-tuned checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="onnx-model/model",
        help="Directory where the ONNX files will be written.",
    )
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version.")
    parser.add_argument(
        "--task",
        default="audio-classification",
        help="Task identifier understood by optimum.exporters.onnx.",
    )
    parser.add_argument(
        "--quantize",
        choices=("dynamic", "none"),
        default="dynamic",
        help=(
            "Apply ONNX Runtime dynamic quantization to keep the model under memory limits. "
            "Use 'none' to skip if quantization causes accuracy regressions."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_export(
        model_name_or_path=args.model_name_or_path,
        output=output_dir,
        opset=args.opset,
        task=args.task,
    )

    model_path = output_dir / "model.onnx"
    if not model_path.exists():
        raise FileNotFoundError(f"Expected ONNX graph at {model_path}, but it was not generated.")

    _quantize_model(model_path, args.quantize)


if __name__ == "__main__":
    main()