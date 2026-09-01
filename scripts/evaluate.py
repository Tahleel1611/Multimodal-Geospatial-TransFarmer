"""Evaluate a trained model with MAE, RMSE, and R2."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from data import ImageDataset, TelemetryDataset, build_aligned_loader
from models import MultimodalRegressor
from utils import regression_metrics, GradCAM, SHAPEnvironmentalImportance

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--checkpoint", default="outputs/model.pt"); p.add_argument("--image-root"); p.add_argument("--telemetry-csv"); p.add_argument("--use-synthetic-telemetry", action="store_true", default=True); p.add_argument("--xai-enabled", action="store_true"); p.add_argument("--batch-size", type=int, default=16); a = p.parse_args()
    images = ImageDataset(a.image_root, train=False); telemetry = TelemetryDataset(a.telemetry_csv, synthetic=a.use_synthetic_telemetry, samples=len(images)); loader = build_aligned_loader(images, telemetry, batch_size=a.batch_size, shuffle=False)
    model = MultimodalRegressor(); checkpoint = Path(a.checkpoint)
    if checkpoint.exists(): model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    pred, target = [], []
    with torch.inference_mode():
        for b in loader: pred.extend(model(b["image"], b["telemetry"], b["trap_count"]).squeeze(1).tolist()); target.extend(b["density"].squeeze(1).tolist())
    print(regression_metrics(pred, target))
    if a.xai_enabled:
        b = next(iter(loader)); print("environment_importance", SHAPEnvironmentalImportance(model)(b["image"][:4], b["telemetry"][:4], b["trap_count"][:4]))

if __name__ == "__main__": main()
