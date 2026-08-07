"""Datamodules for the three shipped examples. All synthetic -- nothing to download.

Delete this file when you delete the examples; nothing in engine/ imports it.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from engine.base import DataModule

# ---------------------------------------------------------------------------
# mlp.py -- y = a random linear map of x, plus noise.
# ---------------------------------------------------------------------------


class SyntheticData(DataModule):
    def __init__(
        self,
        in_dim: int = 8,
        out_dim: int = 1,
        n_train: int = 512,
        n_val: int = 128,
        batch_size: int = 32,
        noise: float = 0.05,
        num_workers: int = 0,
        seed: int = 0,
    ) -> None:
        self.in_dim, self.out_dim = in_dim, out_dim
        self.n_train, self.n_val = n_train, n_val
        self.batch_size, self.noise = batch_size, noise
        self.num_workers, self.seed = num_workers, seed

    def _make(self, n: int, g: torch.Generator) -> TensorDataset:
        x = torch.randn(n, self.in_dim, generator=g)
        y = x @ self._w + self.noise * torch.randn(n, self.out_dim, generator=g)
        return TensorDataset(x, y)

    def setup(self, stage: str) -> None:
        g = torch.Generator().manual_seed(self.seed)
        self._w = torch.randn(self.in_dim, self.out_dim, generator=g)
        self.train_ds = self._make(self.n_train, g)
        self.val_ds = self._make(self.n_val, g)

    def _loader(self, ds: TensorDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=_as_dict,
            drop_last=False,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)


def _as_dict(samples: list[tuple[Tensor, Tensor]]) -> dict[str, Tensor]:
    """Batches are dicts by convention here -- the Trainer never inspects them, so
    any structure works, but dicts keep step methods readable."""
    xs, ys = zip(*samples, strict=True)
    return {"x": torch.stack(xs), "y": torch.stack(ys)}


# ---------------------------------------------------------------------------
# forecast.py -- sin + trend + noise, split temporally.
# ---------------------------------------------------------------------------


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
    """val_dataloader returns a DICT keyed by horizon, so metrics come out namespaced
    val/h8/mae and val/h24/mae."""

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

        # TEMPORAL split. Never shuffle before splitting -- that leaks future into past
        # and the val loss lies to you.
        cut = int(self.length * self.train_frac)
        self.train_series, self.val_series = series[:cut], series[cut:]

        # Scaler fitted on TRAIN ONLY, for the same reason. The model picks it up in
        # on_data_ready and keeps it in buffers.
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


# ---------------------------------------------------------------------------
# pinn.py -- collocation points, no dataset at all.
# ---------------------------------------------------------------------------


class _Collocation(Dataset):
    """Resamples collocation points on every access, so each epoch sees new points."""

    def __init__(self, n: int, x_max: float, fixed: bool = False, seed: int = 0) -> None:
        self.n, self.x_max, self.fixed = n, x_max, fixed
        self.points = torch.rand(n, 1) * x_max if fixed else None

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> Tensor:
        if self.points is not None:
            return self.points[i]
        return torch.rand(1) * self.x_max


class CollocationData(DataModule):
    def __init__(
        self,
        n_train: int = 1024,
        n_val: int = 256,
        x_max: float = 2.0,
        batch_size: int = 128,
        num_workers: int = 0,
    ) -> None:
        self.n_train, self.n_val, self.x_max = n_train, n_val, x_max
        self.batch_size, self.num_workers = batch_size, num_workers

    def setup(self, stage: str) -> None:
        self.train_ds = _Collocation(self.n_train, self.x_max, fixed=False)
        # Validation points are FIXED, so the metric is comparable across evaluations.
        self.val_ds = _Collocation(self.n_val, self.x_max, fixed=True)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            collate_fn=lambda b: {"x": torch.stack(b)},
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)
