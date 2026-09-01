"""Configurable torchvision vision backbones."""
from __future__ import annotations
from typing import Tuple
import torch
from torch import Tensor, nn
try:
    from torchvision import models
except Exception:  # pragma: no cover
    models = None


class TinyVisionBackbone(nn.Module):
    """Dependency-safe CNN fallback with spatial feature maps."""
    def __init__(self, output_dim: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                                 nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                                 nn.Conv2d(64, output_dim, 3, 2, 1), nn.BatchNorm2d(output_dim), nn.ReLU())
        self.output_dim = output_dim
    def forward(self, x: Tensor) -> Tensor: return self.net(x)


class TorchvisionBackbone(nn.Module):
    def __init__(self, name: str, pretrained: bool, output_dim: int) -> None:
        super().__init__()
        if models is None: raise RuntimeError("torchvision is unavailable")
        try:
            if name == "resnet18":
                base = models.resnet18(weights="DEFAULT" if pretrained else None); self.features = nn.Sequential(*list(base.children())[:-2]); source = 512
            elif name in {"mobilenet_v3_small", "mobilenet_v3_large"}:
                base = getattr(models, name)(weights="DEFAULT" if pretrained else None); self.features = base.features; source = base.classifier[0].in_features
            elif name in {"efficientnet_b0", "efficientnet_lite"}:
                base = models.efficientnet_b0(weights="DEFAULT" if pretrained else None); self.features = base.features; source = base.classifier[0].in_features
            else: raise ValueError(f"Unsupported vision backbone: {name}")
        except (RuntimeError, OSError) as exc:
            raise RuntimeError(f"Could not construct {name}: {exc}") from exc
        self.projection = nn.Conv2d(source, output_dim, 1); self.output_dim = output_dim
    def forward(self, x: Tensor) -> Tensor: return self.projection(self.features(x))


def build_vision_backbone(name: str = "mobilenet_v3_small", pretrained: bool = False, output_dim: int = 96) -> nn.Module:
    """Create a named torchvision backbone; use the tiny CNN when unavailable."""
    if name == "tiny" or models is None: return TinyVisionBackbone(output_dim)
    try: return TorchvisionBackbone(name, pretrained, output_dim)
    except (ValueError, RuntimeError): return TinyVisionBackbone(output_dim)
