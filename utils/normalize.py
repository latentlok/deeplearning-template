"""Normalisation as a module, so it rides inside the checkpoint.

The numbers live in BUFFERS. Buffers are part of state_dict(), so they are written
into every checkpoint and restored with the weights. That is the whole point: at
inference you load a checkpoint and the right mean/std arrive with it. Nothing reads
stats.json outside of training.

Defaults are mean 0 / std 1 -- the identity -- so a datamodule that reports no
statistics changes nothing.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class Normalizer(nn.Module):
    """z-score in one direction, back out the other."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    def fit(self, mean: Sequence[float], std: Sequence[float]) -> None:
        """Copy in statistics computed offline. Never computes anything itself."""
        self.mean.copy_(torch.as_tensor(mean, dtype=self.mean.dtype))
        # A constant variable has std 0 and would divide to inf.
        self.std.copy_(torch.as_tensor(std, dtype=self.std.dtype).clamp_min(1e-8))

    def norm(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def denorm(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)
