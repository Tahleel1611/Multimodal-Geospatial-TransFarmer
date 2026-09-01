"""ONNX export and PyTorch/ONNX parity validation."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch


class _ExportWrapper(torch.nn.Module):
    """Expose an unambiguous three-input forward for ONNX tracing."""
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor, telemetry: torch.Tensor, trap_count: torch.Tensor) -> torch.Tensor:
        output = self.model(image, telemetry, trap_count)
        # Export must contain only the scalar regression tensor, never XAI auxiliaries.
        return output[0] if isinstance(output, tuple) else output


def export_onnx(model: torch.nn.Module, path: str | Path, image: torch.Tensor, telemetry: torch.Tensor, trap_count: torch.Tensor, *, opset: int = 17) -> Path:
    """Export a model with dynamic batch and telemetry-time axes."""
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True); model.eval().cpu()
    export_model = _ExportWrapper(model).eval()
    try:
        args = (image.cpu(), telemetry.cpu(), trap_count.cpu())
        torch.onnx.export(export_model, args, str(destination), input_names=["image", "telemetry", "trap_count"], output_names=["pest_density"], dynamic_axes={"image": {0: "batch"}, "telemetry": {0: "batch", 1: "time"}, "trap_count": {0: "batch"}, "pest_density": {0: "batch"}}, opset_version=opset)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        raise RuntimeError("ONNX export failed; install torch.onnx dependencies") from exc
    return destination


def validate_onnx(model: torch.nn.Module, onnx_path: str | Path, image: torch.Tensor, telemetry: torch.Tensor, trap_count: torch.Tensor, *, atol: float = 1e-4) -> float:
    """Compare ONNX Runtime output with PyTorch and return maximum absolute error."""
    try:
        import onnxruntime as ort
    except ImportError as exc: raise RuntimeError("Install onnxruntime to validate ONNX") from exc
    model.eval();
    with torch.inference_mode(): expected = model(image, telemetry, trap_count).cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(["pest_density"], {"image": image.cpu().numpy(), "telemetry": telemetry.cpu().numpy(), "trap_count": trap_count.cpu().numpy()})[0]
    error = float(np.max(np.abs(expected - actual)))
    if error > atol: raise AssertionError(f"ONNX parity error {error:.6g} exceeds atol={atol}")
    return error
