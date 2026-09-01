"""Cross-modal attention where visual spatial tokens query environment context."""
from __future__ import annotations
import torch
from torch import Tensor, nn


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim: int = 96, num_heads: int = 4, dropout: float = .1) -> None:
        super().__init__()
        if embed_dim % num_heads: raise ValueError("embed_dim must be divisible by num_heads")
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, visual_tokens: Tensor, environmental_tokens: Tensor, *, need_weights: bool = False) -> tuple[Tensor, Tensor | None]:
        if visual_tokens.ndim != 3 or environmental_tokens.ndim != 3: raise ValueError("Attention inputs must be [B,N,C]")
        if visual_tokens.shape[0] != environmental_tokens.shape[0] or visual_tokens.shape[2] != environmental_tokens.shape[2]: raise ValueError("Cross-modal batch/channel dimensions must match")
        output, weights = self.attention(visual_tokens, environmental_tokens, environmental_tokens, need_weights=need_weights)
        return self.norm(output + visual_tokens), weights if need_weights else None
