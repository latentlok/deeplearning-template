"""The simplest possible example: a two-layer net.

It exists to prove the wiring end-to-end and to be the file you copy when writing
your own model -- not to be a useful model. Deliberately domain-neutral, and fast
enough to be the smoke test and the contract test.

It also carries the one pattern worth stealing verbatim: data statistics arrive
through on_data_ready and live in BUFFERS. See the comment there.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from engine.base import DataModule, OptimSpec, TaskModule, TrainState


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

        # Normalisation bounds as BUFFERS, not plain attributes: buffers are part of
        # state_dict(), so they are written into every checkpoint and restored with the
        # weights. A model cannot then be loaded for inference against different bounds
        # than it trained on -- which is silent, and ruins the predictions.
        # Defaults are the identity transform, so a datamodule that reports no stats
        # changes nothing.
        self.register_buffer("lower", torch.zeros(in_dim))
        self.register_buffer("upper", torch.ones(in_dim))

    def on_data_ready(self, datamodule: DataModule) -> None:
        """Called once by the Trainer, after datamodule.setup("fit").

        `stats` is a plain dict the datamodule loaded from disk (see utils/stats.py) --
        the model never scans the dataset itself, and eval never calls this at all.
        """
        stats = getattr(datamodule, "stats", None)
        if not stats:
            return
        self.lower.copy_(torch.as_tensor(stats["lower"], dtype=self.lower.dtype))
        self.upper.copy_(torch.as_tensor(stats["upper"], dtype=self.upper.dtype))

    def forward(self, x: Tensor) -> Tensor:
        span = (self.upper - self.lower).clamp_min(1e-8)
        return self.net((x - self.lower) / span)

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
