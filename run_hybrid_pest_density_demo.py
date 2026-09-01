"""End-to-end hybrid UAV-image + synthetic IoT pest-density demonstration.

The script is deliberately self-contained: it first looks for an ImageFolder
dataset (PlantVillage/IP102-style directory), can optionally download CIFAR-10,
and otherwise creates deterministic UAV-like RGB images.  Therefore it can be
run offline and still exercises every model and deployment stage.

Examples
--------
python run_hybrid_pest_density_demo.py
python run_hybrid_pest_density_demo.py --image-root data/PlantVillage
python run_hybrid_pest_density_demo.py --download-cifar --epochs 3

Expected ImageFolder layout is ``root/class_name/image.jpg``.  PlantVillage and
IP102 can both be arranged in that conventional format; their class labels are
not used here because pest density is a regression label synthesised from the
time-aligned IoT signals.
"""
from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torchvision import datasets, transforms
    _TORCHVISION_READY = True
except Exception as exc:  # pragma: no cover - environment-dependent fallback
    datasets = transforms = None
    _TORCHVISION_READY = False
    print(f"torchvision unavailable ({exc}); using procedural images.")


TELEMETRY_STEPS = 24


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_telemetry(index: int, seed: int) -> tuple[np.ndarray, float, float]:
    """Produce a synchronized 24-hour weather sequence, trap count and target.

    Temperature follows a diurnal curve. Humidity is inversely related to heat,
    soil moisture varies slowly, and high humid/warm conditions raise pressure.
    """
    rng = np.random.default_rng(seed + index * 7919)
    hour = np.arange(TELEMETRY_STEPS, dtype=np.float32)
    phase = rng.uniform(-0.7, 0.7)
    temperature = 27 + rng.normal(0, 2.2) + 6 * np.sin((hour - 8) * np.pi / 12 + phase)
    humidity = 72 + rng.normal(0, 5) - 0.9 * (temperature - 27) + 5 * np.sin(hour * np.pi / 12 + phase)
    soil_moisture = 38 + rng.normal(0, 7) + 4 * np.sin((hour - 4) * np.pi / 12 + phase / 2)
    temperature = np.clip(temperature, 12, 45)
    humidity = np.clip(humidity, 25, 99)
    soil_moisture = np.clip(soil_moisture, 5, 85)
    favourable = np.maximum(temperature - 21, 0) * (humidity / 100) * (0.5 + soil_moisture / 200)
    expected_traps = 1.0 + 0.32 * favourable.mean()
    trap_count = float(rng.poisson(expected_traps))
    pest_density = float(max(0, 1.8 * favourable.mean() + 2.7 * trap_count + rng.normal(0, 1.4)))
    telemetry = np.stack([temperature, humidity, soil_moisture], axis=-1).astype(np.float32)
    return telemetry, trap_count, pest_density


class HybridPestDataset(Dataset):
    """Pairs RGB imagery with synchronized synthetic telemetry and density labels."""

    def __init__(self, image_dataset: Optional[Dataset], length: int, seed: int, image_size: int) -> None:
        self.image_dataset, self.length, self.seed, self.image_size = image_dataset, length, seed, image_size
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if _TORCHVISION_READY else None

    def __len__(self) -> int:
        return self.length

    def _procedural_image(self, index: int, density: float) -> Tensor:
        """Offline substitute with crop-like texture and density-linked spots."""
        generator = torch.Generator().manual_seed(self.seed + index)
        size = self.image_size
        image = torch.empty(3, size, size).uniform_(0.05, 0.3, generator=generator)
        image[1] += 0.28  # predominantly green crop canopy
        spots = min(30, max(1, int(density / 2)))
        for _ in range(spots):
            y = int(torch.randint(3, size - 3, (1,), generator=generator))
            x = int(torch.randint(3, size - 3, (1,), generator=generator))
            image[0, y - 2:y + 3, x - 2:x + 3] += 0.55
            image[1, y - 2:y + 3, x - 2:x + 3] -= 0.15
        return image.clamp(0, 1)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        telemetry, traps, density = make_telemetry(index, self.seed)
        if self.image_dataset is not None:
            image, _ = self.image_dataset[index % len(self.image_dataset)]
        else:
            image = self._procedural_image(index, density)
            if self.normalize is not None:
                image = self.normalize(image)
        return {
            "image": image.float(),
            "telemetry": torch.from_numpy(telemetry),
            "trap_count": torch.tensor([traps], dtype=torch.float32),
            "density": torch.tensor([density], dtype=torch.float32),
        }


def load_images(args: argparse.Namespace) -> tuple[Optional[Dataset], str]:
    """Select real local/downloaded images, falling back safely on every error."""
    if not _TORCHVISION_READY:
        return None, "procedural offline images"
    transform = transforms.Compose([transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    if args.image_root:
        try:
            folder = datasets.ImageFolder(args.image_root, transform=transform)
            if len(folder):
                return folder, f"local ImageFolder: {args.image_root} ({len(folder)} images)"
        except Exception as exc:
            print(f"Could not load --image-root ({exc}); trying fallback.")
    if args.download_cifar:
        try:
            cifar = datasets.CIFAR10(args.data_dir, train=True, download=True, transform=transform)
            return cifar, f"downloaded CIFAR-10 placeholder ({len(cifar)} images)"
        except Exception as exc:
            print(f"CIFAR-10 download failed ({exc}); using procedural fallback.")
    return None, "procedural offline images"


class TinyVisionBackbone(nn.Module):
    """Lightweight CNN yielding a 6x6-ish grid of spatial visual tokens."""
    def __init__(self, embedding_dim: int = 96) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, embedding_dim, 3, stride=2, padding=1), nn.BatchNorm2d(embedding_dim), nn.ReLU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class CrossModalAttention(nn.Module):
    """Visual spatial features are Q; environmental/trap tokens are K and V."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.query, self.key, self.value = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.scale = dim ** -0.5

    def forward(self, visual: Tensor, environment: Tensor) -> tuple[Tensor, Tensor]:
        weights = torch.softmax((self.query(visual) @ self.key(environment).transpose(1, 2)) * self.scale, dim=-1)
        return self.out(weights @ self.value(environment)), weights


class TransFarmerPestRegressor(nn.Module):
    def __init__(self, dim: int = 96) -> None:
        super().__init__()
        self.vision = TinyVisionBackbone(dim)
        self.environment_lstm = nn.LSTM(input_size=3, hidden_size=dim, batch_first=True)
        self.trap_encoder = nn.Sequential(nn.Linear(1, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.cross_attention = CrossModalAttention(dim)
        self.regressor = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Dropout(0.1), nn.Linear(dim, 1))

    def forward(self, image: Tensor, telemetry: Tensor, trap_count: Tensor, return_feature_map: bool = False):
        fmap = self.vision(image)
        if return_feature_map:
            fmap.retain_grad()
        visual_tokens = fmap.flatten(2).transpose(1, 2)
        _, (hidden, _) = self.environment_lstm(telemetry)
        environment = torch.stack([hidden[-1], self.trap_encoder(trap_count)], dim=1)
        attended, weights = self.cross_attention(visual_tokens, environment)
        fused = torch.cat([visual_tokens.mean(1), attended.mean(1)], dim=1)
        prediction = self.regressor(fused)
        return (prediction, fmap, weights) if return_feature_map else prediction


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train(); losses = []
    for batch in loader:
        data = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        prediction = model(data["image"], data["telemetry"], data["trap_count"])
        loss = F.mse_loss(prediction, data["density"])
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval(); predictions, targets = [], []
    for batch in loader:
        data = {key: value.to(device) for key, value in batch.items()}
        predictions.extend(model(data["image"], data["telemetry"], data["trap_count"]).squeeze(1).cpu().tolist())
        targets.extend(data["density"].squeeze(1).cpu().tolist())
    pred, true = np.asarray(predictions), np.asarray(targets)
    return {"MAE": float(np.abs(pred - true).mean()), "RMSE": float(np.sqrt(np.mean((pred - true) ** 2))),
            "R2": float(1 - np.sum((pred - true) ** 2) / max(np.sum((true - true.mean()) ** 2), 1e-8))}


def gradcam(model: TransFarmerPestRegressor, sample: dict[str, Tensor], device: torch.device) -> np.ndarray:
    """Return a genuine Grad-CAM heatmap over the last visual feature grid."""
    model.eval(); model.zero_grad(set_to_none=True)
    image = sample["image"].unsqueeze(0).to(device)
    output, fmap, _ = model(image, sample["telemetry"].unsqueeze(0).to(device), sample["trap_count"].unsqueeze(0).to(device), True)
    output.sum().backward()
    cam = (fmap.grad.mean((2, 3), keepdim=True) * fmap).sum(1, keepdim=True).relu()
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam.detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-root", type=str, default=None, help="PlantVillage/IP102 ImageFolder path")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--download-cifar", action="store_true", help="Attempt torchvision CIFAR-10 download")
    parser.add_argument("--epochs", type=int, default=3); parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-samples", type=int, default=128); parser.add_argument("--eval-samples", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=96); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args(); seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, source = load_images(args); print(f"Image source: {source}\nDevice: {device}")
    train_set = HybridPestDataset(images, args.train_samples, args.seed, args.image_size)
    eval_set = HybridPestDataset(images, args.eval_samples, args.seed + 100_000, args.image_size)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_set, batch_size=args.batch_size, num_workers=0)
    model = TransFarmerPestRegressor().to(device); optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs} | MSE: {train_epoch(model, train_loader, optimizer, device):.4f}")
    metrics = evaluate(model, eval_loader, device); print("Evaluation:", ", ".join(f"{name}={value:.3f}" for name, value in metrics.items()))
    sample = eval_set[0]; cam = gradcam(model, sample, device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_dir / "gradcam_attribution.png", cam, cmap="jet")
    print(f"Grad-CAM spatial attribution saved to: {out_dir / 'gradcam_attribution.png'}")
    edge_model = torch.ao.quantization.quantize_dynamic(copy.deepcopy(model).cpu().eval(), {nn.LSTM, nn.Linear}, dtype=torch.qint8)
    edge_batch = next(iter(eval_loader)); start = time.perf_counter()
    with torch.inference_mode():
        edge_prediction = edge_model(edge_batch["image"], edge_batch["telemetry"], edge_batch["trap_count"])
    elapsed_ms = (time.perf_counter() - start) * 1000 / edge_batch["image"].shape[0]
    print(f"INT8 dynamic-quantized edge inference: {elapsed_ms:.2f} ms/frame")
    print(f"Predicted Pest Density (D), first frame: {edge_prediction[0].item():.2f}")
    print("Telemetry preview:\n", pd.DataFrame(sample["telemetry"].numpy(), columns=["temperature_c", "humidity_pct", "soil_moisture_pct"]).head(3))


if __name__ == "__main__":
    main()
