"""The contracts. Two ABCs, and the state/spec dataclasses they exchange.

Deliberately tiny. This fixes only the *boundary* between your model and the Trainer --
every internal is yours. The framework assumes exactly four things:

  1. Training is a loop over batches from a dataloader.
  2. Each step yields a scalar to minimise (or you take manual control).
  3. Parameters are updated by torch.optim optimizers.
  4. Progress is measured in steps.

Anything satisfying those works, anticipated or not.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

# `batch` is never inspected by the Trainer -- it is only moved to the device.
# Dicts, tuples, PyG Batch objects, and anything with a .to() all work. This is what
# lets two-level sampling (pick a geometry, subsample its points) stay a Dataset
# concern rather than a pipeline concern.
type Batch = Any

# Plural on both sides: multi-task / mixture-of-corpora on train; multi-horizon or
# multi-resolution on val. Dict keys become metric namespaces, e.g. "val/h24/mae".
type Loaders = DataLoader | dict[str, DataLoader]


@dataclass
class SchedulerSpec:
    """A scheduler plus the metadata needed to drive it correctly.

    `interval` is not optional in practice: cosine-with-warmup steps per optimizer
    step, classic decay steps per epoch, and ReduceLROnPlateau needs `monitor`. A
    contract without these three fields forces a fork the first time you use warmup.
    """

    scheduler: Any
    interval: str = "step"  # "step" (optimizer steps) | "epoch"
    monitor: str | None = None  # required for ReduceLROnPlateau, e.g. "val/loss"
    frequency: int = 1


@dataclass
class OptimSpec:
    """1..N optimizers. Single-optimizer training is just N=1.

    GANs, actor-critic, and encoder/predictor splits with different LRs all need
    several; a fixed pair would force workarounds.
    """

    optimizers: list[Optimizer]
    schedulers: list[SchedulerSpec] = field(default_factory=list)

    @staticmethod
    def of(optimizer: Optimizer, scheduler: Any = None, **kw: Any) -> OptimSpec:
        """Convenience for the common single-optimizer case."""
        scheds = [SchedulerSpec(scheduler, **kw)] if scheduler is not None else []
        return OptimSpec(optimizers=[optimizer], schedulers=scheds)


@dataclass
class TrainState:
    """Everything needed to resume. Checkpointed alongside weights and optimizers.

    `global_step` counts OPTIMIZER steps, not micro-batches. With grad_accum=4 a step
    is 4 forward/backward passes -- counting micro-batches instead puts every schedule
    and every logged x-axis off by 4x.
    """

    global_step: int = 0
    epoch: int = 0
    samples_seen: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    best_metric: float | None = None
    should_stop: bool = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "samples_seen": self.samples_seen,
            "metrics": dict(self.metrics),
            "best_metric": self.best_metric,
        }

    def load_state_dict(self, sd: dict[str, Any]) -> None:
        self.global_step = sd.get("global_step", 0)
        self.epoch = sd.get("epoch", 0)
        self.samples_seen = sd.get("samples_seen", 0)
        self.metrics = dict(sd.get("metrics", {}))
        self.best_metric = sd.get("best_metric")


class TaskModule(nn.Module, ABC):
    """What you implement. Three methods; everything else is yours.

    A JEPA, a PINN, an FNO, a neural ODE, a Transolver surrogate and a decoder-only LM
    all implement the same three -- the Trainer never learns what domain it is in.
    """

    # True -> you own backward/step entirely. The Trainer's services stay available as
    # helpers (self.trainer.backward / self.trainer.clip_and_step), so taking manual
    # control of one thing does not cost you AMP scaling, grad accumulation, clipping
    # and DDP sync. A ramp, not a cliff.
    manual_optimization: bool = False

    # True -> evaluation is NOT wrapped in torch.no_grad(). Required for PINNs, whose
    # PDE residuals cannot be computed without autograd. The Trainer never uses
    # torch.inference_mode(), which is stricter and taints tensors against re-entering
    # autograd later.
    eval_requires_grad: bool = False

    # Set by the Trainer at fit() time so manual-optimization modules can reach its
    # helpers. Plain attribute, not an nn.Module, so it is not registered as a child.
    trainer: Any = None

    @abstractmethod
    def training_step(self, batch: Batch, state: TrainState) -> dict[str, Tensor]:
        """Return a dict containing "loss" (scalar). Every other entry is logged.

        `state` is passed in so schedules can depend on progress -- teacher-forcing
        ratios, KL annealing, curriculum difficulty, loss-weight ramps.

        Return "batch_size" to control metric weighting. The default infers dim 0,
        which for two-level sampling means *items* (geometries), not points.
        """

    @abstractmethod
    def validation_step(self, batch: Batch, state: TrainState) -> dict[str, Tensor]:
        """Independent of training_step by design.

        This is what lets you train teacher-forced and validate free-running.
        Validating *with* teacher forcing is a classic silent bug: val loss looks
        excellent, the model deploys badly, and nothing errors.
        """

    @abstractmethod
    def configure_optimizers(self) -> OptimSpec: ...

    # -- optional, all no-op by default -------------------------------------------

    def on_data_ready(self, datamodule: DataModule) -> None:
        """Called once in fit(), after datamodule.setup("fit") and before the weights
        move to the device. The one place a model may read from its data.

        This exists for statistics the model must OWN rather than recompute:
        normalisation bounds, a channel mean/std, a vocab size. Copy them into
        buffers here and they ride inside the checkpoint, so inference cannot
        silently use different bounds than training did.

            def on_data_ready(self, dm):
                self.lower.copy_(torch.as_tensor(dm.stats["lower"]))

        NOT called by eval.py -- an evaluation loads its bounds from the checkpoint,
        and recomputing them from whatever data is mounted is exactly the desync this
        prevents. On resume it is called BEFORE the checkpoint loads, so checkpointed
        values win.
        """

    def rollout_step(self, batch: Batch, state: TrainState) -> dict[str, Tensor] | None:
        """Free-running / autoregressive evaluation. Define it and set `rollout_every`.

        Separate from validation_step because it is sequential and dominates
        wall-clock at long horizons -- lopsided for Transformers, where teacher forcing
        is fully parallel but rollout is not. Returns plain scalars like any other step.
        """
        return None

    def save_extra(self, directory: Path) -> None:
        """Persist non-tensor state: tokenizers, vocabs, scaler config.

        safetensors holds tensors only. Without this hook, resume silently pairs
        trained weights with a rebuilt tokenizer -- which mostly works and is
        occasionally catastrophic.
        """

    def load_extra(self, directory: Path) -> None:
        """Inverse of save_extra. Called before the state dict is loaded."""


class DataModule(ABC):
    """Where your data lives. The Trainer only ever calls these."""

    @abstractmethod
    def setup(self, stage: str) -> None:
        """stage is "fit" | "validate" | "test"."""

    @abstractmethod
    def train_dataloader(self) -> Loaders: ...

    @abstractmethod
    def val_dataloader(self) -> Loaders | None: ...

    def state_dict(self) -> dict[str, Any]:
        """Override to make dataloader position resumable.

        Default is empty and the Trainer warns at resume that data position is
        best-effort. For a streaming run this matters: without it, resuming at step 40k
        restarts the iterator from the top and silently retrains the same prefix.
        Solving it properly is genuinely hard, so this does not pretend to.
        """
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None: ...
