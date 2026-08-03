"""The simplest possible example: a two-layer net on random tensors.

It exists to prove the wiring end-to-end and to be the file you read when writing
your own model -- not to be a useful model. Deliberately domain-neutral, and fast
enough to be the smoke test and the contract test.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from dlt.core.base import DataModule, OptimSpec, TaskModule, TrainState


class MLP(TaskModule):
    def __init__(
        self,
        in_dim: int = 8,
        hidden_dim: int = 32,
        out_dim: int = 1,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )
        self.lr, self.weight_decay = lr, weight_decay

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def training_step(self, batch: dict, state: TrainState) -> dict[str, Tensor]:
        loss = nn.functional.mse_loss(self(batch["x"]), batch["y"])
        return {"loss": loss}

    def validation_step(self, batch: dict, state: TrainState) -> dict[str, Tensor]:
        pred = self(batch["x"])
        return {
            "loss": nn.functional.mse_loss(pred, batch["y"]),
            "mae": (pred - batch["y"]).abs().mean(),
        }

    def configure_optimizers(self) -> OptimSpec:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return OptimSpec.of(opt)


class SyntheticData(DataModule):
    """y = a random linear map of x, plus noise. Nothing to download."""

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
