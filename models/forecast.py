"""Windowed timeseries forecasting.

This example exists to get the *timeseries-specific* patterns right, not to be an
interesting model. TCNs and Transformers are plain nn.Modules -- swap `self.net` and
everything here still holds. Its data half lives in dataset/examples.py (SeriesData).

  1. Scaler statistics are nn.Module BUFFERS, fitted on train only and handed over in
     on_data_ready. As buffers they ride along in the checkpoint automatically, so the
     model cannot be loaded against a scaler it did not train with.
  2. Training is teacher-forced (with scheduled sampling), evaluation rolls out
     free-running. Validating *with* teacher forcing is a classic silent bug: val loss
     looks excellent, the model deploys badly, nothing errors.
  3. Multi-horizon evaluation uses the dict-of-loaders contract, so metrics arrive as
     val/h8/mae and val/h24/mae.

Footgun worth knowing: the Trainer resets nothing on your module between batches.
That is exactly what truncated BPTT with carried hidden state needs -- and a silent
bug if you carry state unintentionally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from engine.base import DataModule, OptimSpec, TaskModule, TrainState
from engine.utils import ScheduledValue


class Forecaster(TaskModule):
    """One-step-ahead model applied recursively. window -> next value."""

    def __init__(
        self,
        window: int = 16,
        hidden_dim: int = 64,
        optim: Callable[..., Optimizer] | None = None,
        sched: Callable[..., Any] | None = None,
        lr: float = 1e-3,
        teacher_forcing: ScheduledValue | None = None,
    ) -> None:
        super().__init__()
        self.window = window
        # Factories from configs/optim/ and configs/sched/; `lr` is the fallback for
        # direct construction only.
        self.optim, self.sched, self.lr = optim, sched, lr
        # Default is pure teacher forcing. Anneal it to 0 for scheduled sampling --
        # `-m teacher_forcing.over_steps=500,2000` is why this is an object and not
        # an inline `max(0., 1 - step/2000)`.
        self.teacher_forcing = teacher_forcing or ScheduledValue(1.0, 1.0, 1, "constant")

        self.net = nn.Sequential(nn.Linear(window, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        # Scaler state as buffers -> checkpointed with the weights, cannot desync.
        self.register_buffer("mean", torch.zeros(1))
        self.register_buffer("std", torch.ones(1))

    def set_scaler(self, mean: float, std: float) -> None:
        self.mean.fill_(mean)
        self.std.fill_(max(std, 1e-8))

    def on_data_ready(self, datamodule: DataModule) -> None:
        """Take the scaler the datamodule fitted on the TRAIN split.

        Without this hook the datamodule computes a scaler nobody reads and the model
        normalises with mean=0, std=1 -- a bug that costs accuracy and raises nothing.
        """
        if (scaler := getattr(datamodule, "scaler", None)) is not None:
            self.set_scaler(*scaler)

    def _norm(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def _denorm(self, x: Tensor) -> Tensor:
        return x * self.std + self.mean

    def forward(self, window: Tensor) -> Tensor:
        """(B, window) raw units -> (B,) raw units."""
        return self._denorm(self.net(self._norm(window)).squeeze(-1))

    # -- steps --------------------------------------------------------------------

    def _unroll(self, seq: Tensor, horizon: int, tf_ratio: float) -> Tensor:
        """Roll `horizon` steps. tf_ratio=1 is pure teacher forcing, 0 is free-running.

        Sequential on purpose: it is the same code path for both regimes, so the only
        difference between training and evaluation is the ratio.
        """
        hist = seq[:, : self.window].clone()
        preds = []
        for h in range(horizon):
            nxt = self(hist)
            preds.append(nxt)
            truth = seq[:, self.window + h]
            use_truth = torch.rand_like(truth) < tf_ratio
            fed = torch.where(use_truth, truth, nxt.detach())
            hist = torch.cat([hist[:, 1:], fed.unsqueeze(-1)], dim=1)
        return torch.stack(preds, dim=1)

    def training_step(self, batch: dict, state: TrainState) -> dict[str, Tensor]:
        seq, horizon = batch["seq"], batch["horizon"]
        ratio = self.teacher_forcing(state.global_step)
        pred = self._unroll(seq, int(horizon), ratio)
        target = seq[:, self.window :]
        return {
            "loss": nn.functional.mse_loss(pred, target),
            "teacher_forcing": torch.tensor(ratio),
        }

    def validation_step(self, batch: dict, state: TrainState) -> dict[str, Tensor]:
        """Teacher-forced and cheap -- run every val_every."""
        seq, horizon = batch["seq"], batch["horizon"]
        pred = self._unroll(seq, int(horizon), tf_ratio=1.0)
        target = seq[:, self.window :]
        return {
            "loss": nn.functional.mse_loss(pred, target),
            "mae": (pred - target).abs().mean(),
        }

    def rollout_step(self, batch: dict, state: TrainState) -> dict[str, Tensor]:
        """Free-running -- what deployment actually looks like. Expensive, so it runs
        on the separate `rollout_every` cadence."""
        seq, horizon = batch["seq"], batch["horizon"]
        pred = self._unroll(seq, int(horizon), tf_ratio=0.0)
        target = seq[:, self.window :]
        return {
            "loss": nn.functional.mse_loss(pred, target),
            "mae": (pred - target).abs().mean(),
        }

    def configure_optimizers(self) -> OptimSpec:
        opt = (
            self.optim(self.parameters())
            if self.optim is not None
            else torch.optim.AdamW(self.parameters(), lr=self.lr)
        )
        sched = self.sched(opt) if self.sched is not None else None
        return OptimSpec.of(opt, sched, interval="step")
