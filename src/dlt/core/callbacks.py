"""Callbacks OBSERVE; they never own the update.

Checkpointing, early stopping, LR logging and gradient statistics are side-effects.
The optimisation logic stays in the Trainer and the TaskModule, so no callback can
silently change your results.

The consequence, stated honestly rather than sold as a virtue: mixup, cutmix and
adversarial perturbation cannot be callbacks. They belong in collate_fn or in your
training_step. Likewise EMA of a target encoder -- for JEPA-style methods that update
*is* the algorithm, and hiding it in a hook is exactly what this rule prevents.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import torch

from dlt.core.base import TaskModule, TrainState
from dlt.core.checkpoint import save_checkpoint

log = logging.getLogger(__name__)


class Callback:
    """Every hook is a no-op, so you override only what you need."""

    def on_fit_start(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None: ...
    def on_fit_end(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None: ...
    def on_epoch_end(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None: ...
    def on_before_optimizer_step(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None:
        """Fires while gradients still exist. Use this, not on_train_batch_end, for
        anything that reads .grad -- clip_and_step zeroes them straight after."""

    def on_train_batch_end(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None: ...
    def on_val_end(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None: ...
    def on_exception(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None: ...


class Checkpoint(Callback):
    """Writes `last/` every save_every steps, and `best/` whenever the monitored
    metric improves."""

    def __init__(
        self,
        dirpath: str | Path = "ckpt",
        monitor: str | None = None,
        save_every: int = 0,
        save_last: bool = True,
        save_best: bool = True,
        fmt: Any = "safetensors",
    ) -> None:
        self.dirpath = Path(dirpath)
        self.monitor, self.save_every = monitor, save_every
        self.save_last, self.save_best, self.fmt = save_last, save_best, fmt

    def _save(self, trainer: Any, module: TaskModule, state: TrainState, tag: str) -> None:
        save_checkpoint(
            self.dirpath / tag,
            module,
            optimizers=trainer.optimizers,
            schedulers=trainer.schedulers,
            state=state,
            datamodule=getattr(trainer, "datamodule", None),
            fmt=self.fmt,
        )

    def on_train_batch_end(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None:
        if self.save_every and state.global_step % self.save_every == 0 and self.save_last:
            self._save(trainer, module, state, "last")

    def on_val_end(self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any) -> None:
        key = self.monitor or trainer.monitor
        value = state.metrics.get(key)
        if value is None or not self.save_best:
            return
        if trainer.is_better(value):
            state.best_metric = value
            self._save(trainer, module, state, "best")
            log.info("new best %s=%.6g at step %d", key, value, state.global_step)

    def on_fit_end(self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any) -> None:
        if self.save_last:
            self._save(trainer, module, state, "last")


class EarlyStopping(Callback):
    def __init__(
        self, monitor: str | None = None, patience: int = 10, min_delta: float = 0.0
    ) -> None:
        self.monitor, self.patience, self.min_delta = monitor, patience, min_delta
        self.best: float | None = None
        self.waited = 0

    def on_val_end(self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any) -> None:
        key = self.monitor or trainer.monitor
        value = state.metrics.get(key)
        if value is None:
            return
        improved = self.best is None or (
            value < self.best - self.min_delta
            if trainer.monitor_mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best, self.waited = value, 0
        else:
            self.waited += 1
            if self.waited >= self.patience:
                log.info("early stop: %s did not improve for %d evals", key, self.patience)
                state.should_stop = True


class LRMonitor(Callback):
    """Logs the LR of every param group of every optimizer."""

    def on_train_batch_end(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None:
        if not trainer.log_every or state.global_step % trainer.log_every:
            return
        lrs = {}
        for i, opt in enumerate(trainer.optimizers):
            for j, g in enumerate(opt.param_groups):
                name = f"lr/opt{i}" if len(opt.param_groups) == 1 else f"lr/opt{i}_g{j}"
                lrs[name] = g["lr"]
        trainer._log(lrs, state.global_step)


class GradStats(Callback):
    """Weight and gradient histograms plus a global grad norm.

    Expensive, hence `every`. Complex parameters are split into abs/real/imag by the
    TensorBoard logger rather than passed through -- add_histogram does not reject
    complex, it silently discards the imaginary part.

    Runs on on_before_optimizer_step, not on_train_batch_end: clip_and_step ends with
    zero_grad(set_to_none=True), so by the end of the batch every .grad is None and
    this would silently report a global norm of 0.0 forever.

    In manual_optimization the module does its own stepping, so gradients may already
    be cleared by the time this fires -- read them in your own training_step instead.
    """

    def __init__(self, every: int = 500, histograms: bool = True) -> None:
        self.every, self.histograms = every, histograms

    def on_before_optimizer_step(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None:
        if not self.every or state.global_step % self.every:
            return
        total = 0.0
        for name, p in module.named_parameters():
            if p.grad is None:
                continue
            total += float(p.grad.detach().float().norm() ** 2)
            if self.histograms and trainer.logger is not None:
                trainer.logger.log_histogram(f"weights/{name}", p.detach(), state.global_step)
                trainer.logger.log_histogram(f"grads/{name}", p.grad.detach(), state.global_step)
        trainer._log({"grad/global_norm": total**0.5}, state.global_step)


class Timer(Callback):
    """Wall-clock and throughput. Cheap, and the first thing you want when a run is
    slower than expected."""

    def on_fit_start(self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any) -> None:
        self.t0 = time.perf_counter()

    def on_fit_end(self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any) -> None:
        dt = time.perf_counter() - getattr(self, "t0", time.perf_counter())
        log.info(
            "finished in %.1fs | %d steps | %d samples | %.1f samples/s",
            dt,
            state.global_step,
            state.samples_seen,
            state.samples_seen / max(dt, 1e-9),
        )


class NaNGuard(Callback):
    """Stop on a non-finite loss instead of burning hours producing NaNs."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def on_train_batch_end(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None:
        if not self.enabled:
            return
        loss = state.metrics.get("train/loss")
        if loss is not None and not torch.isfinite(torch.tensor(loss)):
            log.error("non-finite loss at step %d -- stopping", state.global_step)
            state.should_stop = True
