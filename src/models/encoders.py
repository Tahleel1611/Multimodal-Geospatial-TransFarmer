"""Environmental temporal and scalar encoders."""
from __future__ import annotations
import torch
from torch import Tensor, nn


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim: int = 3, hidden_dim: int = 96, kind: str = "lstm", layers: int = 1) -> None:
        super().__init__(); self.kind = kind
        if kind == "lstm": self.encoder = nn.LSTM(input_dim, hidden_dim, layers, batch_first=True)
        elif kind in {"transformer", "1d_transformer"}:
            self.input = nn.Linear(input_dim, hidden_dim); layer = nn.TransformerEncoderLayer(hidden_dim, 4, batch_first=True, norm_first=True); self.encoder = nn.TransformerEncoder(layer, layers)
        else: raise ValueError("kind must be 'lstm' or 'transformer'")
        self.hidden_dim = hidden_dim
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3: raise ValueError(f"Telemetry must be [B,T,F], got {tuple(x.shape)}")
        return self.encoder(x)[0] if self.kind == "lstm" else self.encoder(self.input(x))


class TrapCountEncoder(nn.Module):
    def __init__(self, output_dim: int = 96) -> None:
        super().__init__(); self.net = nn.Sequential(nn.Linear(1, output_dim), nn.LayerNorm(output_dim), nn.GELU())
    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1: x = x.unsqueeze(-1)
        if x.ndim != 2 or x.shape[1] != 1: raise ValueError("trap_count must be [B,1]")
        return self.net(x)
