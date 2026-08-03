"""Windowed timeseries forecasting.

This example exists to get the *timeseries-specific* patterns right, not to be an
interesting model. TCNs and Transformers are plain nn.Modules -- swap `self.net` and
everything here still holds. What actually bites in timeseries is data and state:

  1. Scaler statistics are nn.Module BUFFERS. Fitted on train only, they must match
     the weights at inference; as buffers they ride along in the checkpoint
     automatically and cannot desync.
  2. The split is TEMPORAL, never random. A shuffled split leaks future into past and
     the val loss lies to you.
  3. Training is teacher-forced (with scheduled sampling), evaluation rolls out
     free-running. Validating *with* teacher forcing is a classic silent bug: val loss
     looks excellent, the model deploys badly, nothing errors.
  4. Multi-horizon evaluation uses the dict-of-loaders contract, so metrics arrive as
     val/h8/mae and val/h24/mae.

Footgun worth knowing: the Trainer resets nothing on your module between batches.
That is exactly what truncated BPTT with carried hidden state needs -- and a silent
bug if you carry state unintentionally.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from dlt.core.base import DataModule, OptimSpec, TaskModule, TrainState
from dlt.core.utils import ScheduledValue


class Forecaster(TaskModule):
    """One-step-ahead model applied recursively. window -> next value."""

    def __init__(
        self,
        window: int = 16,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        teacher_forcing: ScheduledValue | None = None,
    ) -> None:
        super().__init__()
        self.window = window
        self.lr = lr
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
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
        return OptimSpec.of(opt, sched, interval="step")


class _Windows(Dataset):
    def __init__(self, series: Tensor, window: int, horizon: int) -> None:
        self.series, self.window, self.horizon = series, window, horizon
        self.n = len(series) - window - horizon + 1
        if self.n <= 0:
            raise ValueError(
                f"series of length {len(series)} is too short for window={window} horizon={horizon}"
            )

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Tensor:
        return self.series[i : i + self.window + self.horizon]


class SeriesData(DataModule):
    """Synthetic series: sin + trend + noise. Nothing to download.

    val_dataloader returns a DICT keyed by horizon, so metrics come out namespaced
    val/h8/mae and val/h24/mae.
    """

    def __init__(
        self,
        length: int = 2000,
        window: int = 16,
        horizons: tuple[int, ...] = (8, 24),
        train_frac: float = 0.7,
        batch_size: int = 32,
        noise: float = 0.05,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        self.length, self.window, self.horizons = length, window, tuple(horizons)
        self.train_frac, self.batch_size = train_frac, batch_size
        self.noise, self.num_workers, self.seed = noise, num_workers, seed
        self.scaler: tuple[float, float] = (0.0, 1.0)

    def setup(self, stage: str) -> None:
        g = torch.Generator().manual_seed(self.seed)
        t = torch.arange(self.length, dtype=torch.get_default_dtype())
        series = torch.sin(t * 0.1) + 0.3 * torch.sin(t * 0.031) + 0.0005 * t
        series = series + self.noise * torch.randn(self.length, generator=g)

        # TEMPORAL split. Never shuffle before splitting -- that leaks future into past.
        cut = int(self.length * self.train_frac)
        self.train_series, self.val_series = series[:cut], series[cut:]

        # Scaler fitted on TRAIN ONLY, for the same reason.
        self.scaler = (float(self.train_series.mean()), float(self.train_series.std()))

        self.train_horizon = min(self.horizons)
        self.train_ds = _Windows(self.train_series, self.window, self.train_horizon)
        self.val_ds = {h: _Windows(self.val_series, self.window, h) for h in self.horizons}

    def _loader(self, ds: Dataset, horizon: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=lambda b: {"seq": torch.stack(b), "horizon": horizon},
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, self.train_horizon, shuffle=True)

    def val_dataloader(self) -> dict[str, DataLoader]:
        # Validation is never shuffled -- order is meaningful here.
        return {f"h{h}": self._loader(ds, h, shuffle=False) for h, ds in self.val_ds.items()}
