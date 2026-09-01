"""Train the modular TransFarmer regressor."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from data import ImageDataset, TelemetryDataset, build_aligned_loader
from models import MultimodalRegressor

def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--image-root"); p.add_argument("--telemetry-csv"); p.add_argument("--use-synthetic-telemetry", action="store_true", default=True); p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=16); p.add_argument("--learning-rate", type=float, default=1e-3); p.add_argument("--quantize", action="store_true"); p.add_argument("--export-onnx", action="store_true"); p.add_argument("--xai-enabled", action="store_true"); p.add_argument("--output-dir", default="outputs")
    a = p.parse_args(); torch.manual_seed(42); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = ImageDataset(a.image_root); telemetry = TelemetryDataset(a.telemetry_csv, synthetic=a.use_synthetic_telemetry, samples=len(images)); loader = build_aligned_loader(images, telemetry, batch_size=a.batch_size)
    model = MultimodalRegressor().to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=a.learning_rate)
    for epoch in range(a.epochs):
        model.train(); total = 0.0
        for b in loader:
            x = {k: v.to(device) for k, v in b.items() if isinstance(v, torch.Tensor)}; optimizer.zero_grad(); loss = F.mse_loss(model(x["image"], x["telemetry"], x["trap_count"]), x["density"]); loss.backward(); optimizer.step(); total += loss.item()
        print(f"epoch={epoch + 1} mse={total / max(1, len(loader)):.4f}")
    Path(a.output_dir).mkdir(parents=True, exist_ok=True); torch.save(model.state_dict(), Path(a.output_dir) / "model.pt")
    if a.export_onnx:
        from export import export_onnx
        b = next(iter(loader)); export_onnx(model, Path(a.output_dir) / "transfarmer.onnx", b["image"], b["telemetry"], b["trap_count"])

if __name__ == "__main__": main()
