"""CLI wrapper for ONNX export and optional parity validation."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from data import ImageDataset, TelemetryDataset, build_aligned_loader
from models import MultimodalRegressor
from export import export_onnx, validate_onnx

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", default="outputs/model.pt"); p.add_argument("--output", default="outputs/transfarmer.onnx"); p.add_argument("--validate", action="store_true"); a = p.parse_args()
    model = MultimodalRegressor(); checkpoint = Path(a.checkpoint)
    if checkpoint.exists(): model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    images, telemetry = ImageDataset(length=2, train=False), TelemetryDataset(samples=2); b = next(iter(build_aligned_loader(images, telemetry, batch_size=2, shuffle=False)))
    path = export_onnx(model, a.output, b["image"], b["telemetry"], b["trap_count"]); print(path)
    if a.validate: print("max_abs_error", validate_onnx(model, path, b["image"], b["telemetry"], b["trap_count"]))

if __name__ == "__main__": main()
