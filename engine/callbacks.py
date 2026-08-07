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
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from engine.base import TaskModule, TrainState
from engine.checkpoint import save_checkpoint
from engine.utils import is_rank_zero

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
    """Four kinds of checkpoint, because they answer different questions.

        ckpt/last/            rolling resume point, overwritten every `last_every`
        ckpt/step_00001000/   permanent snapshot every `every_steps`, pruned to `keep_last`
        ckpt/best/            full state at the best monitored metric -- resume from here
        ckpt/best_weights/    the same weights with NO optimizer state -- ship this

    `best/` and `best_weights/` are both written because they are different artifacts:
    one continues training, the other is what you load for inference. Keeping only the
    full one means every deployment drags optimizer moments around; keeping only the
    weights means a crashed run cannot pick up from its best point.

    Snapshots are step-tagged and permanent so a divergence at step 40k can be traced
    back through steps that `last/` has long since overwritten.
    """

    def __init__(
        self,
        dirpath: str | Path = "ckpt",
        monitor: str | None = None,
        every_steps: int = 0,
        keep_last: int = 3,
        last_every: int = 500,
        save_last: bool = True,
        save_best: bool = True,
        save_best_weights: bool = True,
        fmt: Any = "safetensors",
    ) -> None:
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.every_steps, self.keep_last = every_steps, keep_last
        self.last_every, self.save_last = last_every, save_last
        self.save_best, self.save_best_weights = save_best, save_best_weights
        self.fmt = fmt

    def _save(
        self,
        trainer: Any,
        module: TaskModule,
        state: TrainState,
        tag: str,
        weights_only: bool = False,
    ) -> None:
        save_checkpoint(
            self.dirpath / tag,
            module,
            optimizers=trainer.optimizers,
            schedulers=trainer.schedulers,
            state=state,
            datamodule=getattr(trainer, "datamodule", None),
            fmt=self.fmt,
            weights_only=weights_only,
        )

    def _prune(self) -> None:
        """Keep the newest `keep_last` step snapshots. keep_last=0 keeps every one.

        Sorted by name, which is why the step is zero-padded -- lexical order and
        numeric order must not disagree at step 10000.
        """
        if not self.keep_last or not is_rank_zero():
            return
        snaps = sorted(p for p in self.dirpath.glob("step_*") if p.is_dir())
        for old in snaps[: -self.keep_last]:
            shutil.rmtree(old, ignore_errors=True)
            log.info("pruned old checkpoint %s", old.name)

    def on_train_batch_end(
        self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any
    ) -> None:
        step = state.global_step
        if self.save_last and self.last_every and step % self.last_every == 0:
            self._save(trainer, module, state, "last")
        if self.every_steps and step % self.every_steps == 0:
            self._save(trainer, module, state, f"step_{step:08d}")
            self._prune()

    def on_val_end(self, trainer: Any, module: TaskModule, state: TrainState, **kw: Any) -> None:
        key = self.monitor or trainer.monitor
        value = state.metrics.get(key)
        if value is None or not (self.save_best or self.save_best_weights):
            return
        if trainer.is_better(value):
            state.best_metric = value
            if self.save_best:
                self._save(trainer, module, state, "best")
            if self.save_best_weights:
                self._save(trainer, module, state, "best_weights", weights_only=True)
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
            # .float() on a complex grad discards the imaginary part, so the norm
            # would silently be the real part's. abs() first keeps the magnitude.
            g = p.grad.detach()
            total += float((g.abs() if g.is_complex() else g).float().norm() ** 2)
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
