"""Candidate-aware sequence mixer for implicit-feedback recommendation."""

import math

import torch
from torch import nn


class MixFormer(nn.Module):
    """Combine global self-attention, local convolution, and candidate attention."""

    def __init__(self, num_items, dim=64, heads=4, dropout=0.1, max_history=200):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.item_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_history + 1, dim, padding_idx=0)
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.local_mixer = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.gate = nn.Linear(dim * 2, dim)
        self.output = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, 1))
        self.dropout = nn.Dropout(dropout)
        self.dim = dim
        self.max_history = max_history

    def forward(self, history, candidate):
        if history.ndim != 2 or candidate.ndim != 1 or history.size(0) != candidate.size(0):
            raise ValueError("history must be [batch, time] and candidate [batch]")
        if history.size(1) > self.max_history:
            raise ValueError("history exceeds max_history")
        valid = history.ne(0)
        if not torch.all(valid.any(dim=1)):
            raise ValueError("each history needs at least one item")
        positions = valid.long().cumsum(dim=1) * valid.long()
        x = self.item_embedding(history) + self.position_embedding(positions)
        global_x, _ = self.self_attention(x, x, x, key_padding_mask=~valid, need_weights=False)
        local_x = self.local_mixer(x.transpose(1, 2)).transpose(1, 2)
        gate = torch.sigmoid(self.gate(torch.cat((global_x, local_x), dim=-1)))
        mixed = self.dropout((gate * global_x + (1 - gate) * local_x) * valid.unsqueeze(-1))
        query = self.item_embedding(candidate)
        scores = torch.einsum("bd,btd->bt", query, mixed) / math.sqrt(self.dim)
        weights = torch.softmax(scores.masked_fill(~valid, torch.finfo(scores.dtype).min), dim=1)
        context = torch.einsum("bt,btd->bd", weights, mixed)
        return self.output(torch.cat((query, context), dim=-1)).squeeze(-1)
