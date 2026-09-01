"""Multimodal pest-density regression model."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from .backbones import build_vision_backbone
from .encoders import TemporalEncoder, TrapCountEncoder
from .fusion import CrossModalAttention


class MultimodalRegressor(nn.Module):
    def __init__(self, *, backbone: str = "mobilenet_v3_small", pretrained: bool = False, feature_dim: int = 96,
                 telemetry_features: int = 3, temporal_encoder: str = "lstm", num_heads: int = 4, dropout: float = .15) -> None:
        super().__init__(); self.vision = build_vision_backbone(backbone, pretrained, feature_dim)
        self.temporal = TemporalEncoder(telemetry_features, feature_dim, temporal_encoder); self.trap = TrapCountEncoder(feature_dim)
        self.fusion = CrossModalAttention(feature_dim, num_heads, dropout)
        self.head = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(feature_dim, 1))
    def forward(self, image: Tensor, telemetry: Tensor, trap_count: Tensor, *, return_attention: bool = False):
        fmap = self.vision(image); assert fmap.ndim == 4 and fmap.shape[1] == self.temporal.hidden_dim
        visual = fmap.flatten(2).transpose(1, 2); env = self.temporal(telemetry); trap = self.trap(trap_count).unsqueeze(1)
        context = torch.cat([env, trap], dim=1); attended, weights = self.fusion(visual, context, need_weights=return_attention)
        prediction = self.head(torch.cat([visual.mean(1), attended.mean(1)], dim=1)); assert prediction.shape == (image.shape[0], 1)
        return (prediction, fmap, weights) if return_attention else prediction
