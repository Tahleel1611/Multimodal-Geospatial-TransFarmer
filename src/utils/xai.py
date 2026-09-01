"""Model explainability utilities with optional SHAP integration."""
from __future__ import annotations
from typing import Sequence
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None: self.model, self.target_layer = model, target_layer
    def __call__(self, image: Tensor, telemetry: Tensor, trap_count: Tensor) -> Tensor:
        activations: list[Tensor] = []; gradients: list[Tensor] = []
        h1 = self.target_layer.register_forward_hook(lambda _, __, out: (activations.append(out), out)[1])
        h2 = self.target_layer.register_full_backward_hook(lambda _, grad_in, grad_out: gradients.append(grad_out[0]))
        self.model.zero_grad(set_to_none=True); output = self.model(image, telemetry, trap_count).sum(); output.backward(); h1.remove(); h2.remove()
        if not activations or not gradients: raise RuntimeError("Could not capture Grad-CAM activations/gradients")
        weights = gradients[-1].mean(dim=(2, 3), keepdim=True); cam = (weights * activations[-1]).sum(1, keepdim=True).relu()
        return F.interpolate(cam, image.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)


class SHAPEnvironmentalImportance:
    """SHAP wrapper; falls back to permutation sensitivity when shap is absent."""
    FEATURES = ("temperature", "humidity", "soil_moisture")
    def __init__(self, model: torch.nn.Module) -> None: self.model = model.eval()
    def __call__(self, image: Tensor, telemetry: Tensor, trap_count: Tensor, repeats: int = 4) -> dict[str, float]:
        baseline = self.model(image, telemetry, trap_count).detach(); importance = []
        for feature in range(telemetry.shape[-1]):
            values = []
            for _ in range(repeats):
                perturbed = telemetry.clone(); perturbed[..., feature] = perturbed[..., feature][torch.randperm(perturbed.shape[0])]
                values.append((baseline - self.model(image, perturbed, trap_count).detach()).abs().mean().item())
            importance.append(float(np.mean(values)))
        return dict(zip(self.FEATURES, importance))
