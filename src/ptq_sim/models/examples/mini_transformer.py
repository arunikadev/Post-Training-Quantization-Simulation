"""MiniTransformerClassifier using explicit nn.Linear projections for transparent PTQ."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ptq_sim.models.registry import register_model


class ExplicitSelfAttention(nn.Module):
    """Multi-head self-attention built from explicit nn.Linear Q/K/V/Out projections."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads}).")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_probs = F.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_probs, v)
        context = context.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(context)


class TransformerBlock(nn.Module):
    """Pre-LN Transformer block with explicit Linear attention and FFN layers."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ExplicitSelfAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.fc2(F.gelu(self.fc1(self.norm2(x))))
        return x


@register_model("mini_transformer")
class MiniTransformerClassifier(nn.Module):
    """Compact Transformer sequence classifier whose Linear projections quantize cleanly."""

    def __init__(
        self,
        vocab_size: int = 128,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 2,
        d_ff: int = 128,
        num_classes: int = 4,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model=d_model, n_heads=n_heads, d_ff=d_ff) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        seq_len = x.shape[1]
        h = self.token_emb(x) + self.pos_emb[:, :seq_len, :]
        for block in self.blocks:
            h = block(h)
        h = self.norm(h)
        pooled = h.mean(dim=1)
        return self.head(pooled)
