"""Fast shape-contract tests for the modular data and model components."""
from __future__ import annotations
import sys
from pathlib import Path
import torch
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data import ImageDataset, TelemetryDataset, align_sliding_windows, build_aligned_loader
from models import CrossModalAttention, MultimodalRegressor, TemporalEncoder


def test_synthetic_loader_shapes_across_batches() -> None:
    images = ImageDataset(length=5, image_size=64)
    telemetry = TelemetryDataset(window=24, samples=5, synthetic=True)
    loader = build_aligned_loader(images, telemetry, batch_size=2, shuffle=False)
    for batch in loader:
        assert batch["image"].ndim == 4
        assert batch["telemetry"].shape[1:] == (24, 3)
        assert batch["trap_count"].shape == (batch["image"].shape[0], 1)


def test_forward_passes_multiple_batch_sizes() -> None:
    model = MultimodalRegressor(backbone="tiny", feature_dim=96)
    for batch_size in (1, 2, 7):
        output = model(torch.randn(batch_size, 3, 64, 64), torch.randn(batch_size, 24, 3), torch.randn(batch_size, 1))
        assert output.shape == (batch_size, 1)


def test_fusion_shape_assertions_and_temporal_transformer() -> None:
    encoder = TemporalEncoder(kind="transformer")
    assert encoder(torch.randn(3, 24, 3)).shape == (3, 24, 96)
    fusion = CrossModalAttention(96, 4)
    output, weights = fusion(torch.randn(2, 16, 96), torch.randn(2, 25, 96), need_weights=True)
    assert output.shape == (2, 16, 96)
    assert weights is not None and weights.shape == (2, 16, 25)


def test_timestamp_alignment_has_no_lookahead() -> None:
    telemetry = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=5, freq="h"), "temperature_c": range(5)})
    windows = align_sliding_windows([pd.Timestamp("2026-01-01 04:00")], telemetry, window=3)
    assert len(windows[0]) == 3 and windows[0]["timestamp"].iloc[-1] == pd.Timestamp("2026-01-01 04:00")
