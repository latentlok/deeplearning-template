"""The simplest possible example: a two-layer net.

It exists to prove the wiring end-to-end and to be the file you copy when writing your
own model -- not to be a useful model. Deliberately domain-neutral, and fast enough to
be the smoke test and the contract test.

It also carries the one pattern worth stealing verbatim: NORMALISATION LIVES HERE, in
buffers, not in the dataloader.

    forward(x)  takes raw units in and gives raw units back.

Inside, x is normalised, the net runs in normalised space, and the prediction is
denormalised on the way out. Because the statistics are buffers on this module, they
are in state_dict(), so they are written into every checkpoint and restored with the
weights. Inference therefore needs the checkpoint and nothing else -- no stats.json, no
dataloader, no chance of normalising against different numbers than training used.

Loss is computed in NORMALISED space, so every target channel contributes on the same
scale regardless of its units. `mae` is reported in raw units, because that is the
number anyone reading a metric actually wants.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from engine.base import DataModule, OptimSpec, TaskModule, TrainState
from utils.normalize import Normalizer


class MLP(TaskModule):
    def __init__(
        self,
        in_dim: int = 8,
        hidden_dim: int = 32,
        out_dim: int = 1,
        optim: Callable[..., Optimizer] | None = None,
        sched: Callable[..., Any] | None = None,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim)
        )
        # Factories from configs/optim/ and configs/sched/ -- partials still waiting
        # for the thing they act on. `lr` is only the fallback for direct construction
        # (MLP(in_dim=2) in a test or a notebook); through Hydra the one source of
        # truth for the learning rate is configs/optim/.
        self.optim, self.sched, self.lr = optim, sched, lr

        # Both directions. Defaults are the identity, so a datamodule that reports no
        # statistics (the synthetic examples) changes nothing.
        self.x_norm = Normalizer(in_dim)
        self.y_norm = Normalizer(out_dim)

    def on_data_ready(self, datamodule: DataModule) -> None:
        """Called once by the Trainer, after datamodule.setup("fit").

        `stats` is a plain dict the datamodule assembled from stats.json -- the model
        never scans the dataset itself, and eval never calls this at all: an evaluation
        takes its statistics from the checkpoint, not from whatever data is mounted.
        """
        stats = getattr(datamodule, "stats", None)
        if not stats:
            return
        self.x_norm.fit(stats["x_mean"], stats["x_std"])
        self.y_norm.fit(stats["y_mean"], stats["y_std"])

    def forward(self, x: Tensor) -> Tensor:
        """Raw units in, raw units out. This is what inference calls."""
        return self.y_norm.denorm(self.net(self.x_norm.norm(x)))

    def training_step(self, batch: dict, state: TrainState) -> dict[str, Tensor]:
        pred = self.net(self.x_norm.norm(batch["x"]))
        loss = nn.functional.mse_loss(pred, self.y_norm.norm(batch["y"]))
        return {"loss": loss}

    def validation_step(self, batch: dict, state: TrainState) -> dict[str, Tensor]:
        pred = self.net(self.x_norm.norm(batch["x"]))
        return {
            # Normalised, so it is comparable across runs and across targets.
            "loss": nn.functional.mse_loss(pred, self.y_norm.norm(batch["y"])),
            # Raw units, so it means something to a human.
            "mae": (self.y_norm.denorm(pred) - batch["y"]).abs().mean(),
        }

    def configure_optimizers(self) -> OptimSpec:
        opt = (
            self.optim(self.parameters())
            if self.optim is not None
            else torch.optim.AdamW(self.parameters(), lr=self.lr)
        )
        sched = self.sched(opt) if self.sched is not None else None
        # interval="step": schedulers here count optimizer steps, matching global_step.
        return OptimSpec.of(opt, sched, interval="step")
