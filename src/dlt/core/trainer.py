"""The training loop. One concrete class with small overridable methods.

Subclass and override ONE method to make a new trainer -- no ABC hierarchy, which
would force every trainer to reimplement the skeleton.

The loop is STEP-FIRST. Epochs are a derived convenience and `len(dataloader)` is
never called, so streaming / IterableDataset training works without a fake epoch.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel

from dlt.core.base import DataModule, OptimSpec, TaskModule, TrainState
from dlt.core.utils import (
    MetricAccumulator,
    effective_batch_size,
    get_world_size,
    infer_batch_size,
    is_rank_zero,
    move_to_device,
)

log = logging.getLogger(__name__)


class _BatchStream:
    """Cycles one or more loaders forever, counting epoch wraps. Never calls len().

    Given a dict of loaders it yields a dict of batches per step, cycling each
    independently -- the multi-task / mixture-of-corpora / multi-view case.
    """

    def __init__(self, loaders: Any) -> None:
        self._dict = isinstance(loaders, dict)
        self._loaders: dict[str, Any] = loaders if self._dict else {"": loaders}
        self._iters: dict[str, Iterator] = {k: iter(v) for k, v in self._loaders.items()}
        self.epoch = 0
        self.epoch_just_ended = False

    def next(self) -> Any:
        self.epoch_just_ended = False
        out = {}
        for name, loader in self._loaders.items():
            try:
                out[name] = next(self._iters[name])
            except StopIteration:
                self._iters[name] = iter(loader)
                out[name] = next(self._iters[name])
                self.epoch += 1
                self.epoch_just_ended = True
        return out if self._dict else out[""]


class Trainer:
    def __init__(
        self,
        max_steps: int = 1000,
        max_epochs: int | None = None,
        grad_accum: int = 1,
        clip_grad: float | None = None,
        amp_dtype: torch.dtype | None = None,
        param_dtype: torch.dtype = torch.float32,
        compile: bool = False,
        compile_mode: str = "default",
        device: str = "auto",
        log_every: int = 50,
        val_every: int = 500,
        rollout_every: int = 0,
        monitor: str = "val/loss",
        monitor_mode: str = "min",
        callbacks: list[Any] | None = None,
        logger: Any = None,
    ) -> None:
        self.max_steps, self.max_epochs = max_steps, max_epochs
        self.grad_accum = max(1, grad_accum)
        self.clip_grad = clip_grad
        self.amp_dtype, self.param_dtype = amp_dtype, param_dtype
        self.compile, self.compile_mode = compile, compile_mode
        self.log_every, self.val_every, self.rollout_every = log_every, val_every, rollout_every
        self.monitor, self.monitor_mode = monitor, monitor_mode
        self.callbacks = list(callbacks or [])
        self.logger = logger

        self.device = torch.device(
            ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        )
        # GradScaler is only meaningful for fp16; bf16 has fp32's exponent range.
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(amp_dtype is torch.float16))
        self.module: Any = None
        self.datamodule: DataModule | None = None
        self.optimizers: list[Any] = []
        self.schedulers: list[Any] = []
        self.state = TrainState()
        self._last_val_step = -1

    # -- services, callable from manual_optimization too ---------------------------
    #
    # Exposed so that taking manual control of ONE thing does not cost you AMP
    # scaling, gradient clipping and DDP sync all at once. A ramp, not a cliff.

    def autocast(self) -> Any:
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.amp.autocast(self.device.type, dtype=self.amp_dtype)

    def backward(self, loss: torch.Tensor) -> None:
        self.scaler.scale(loss).backward()

    def clip_and_step(self, optimizer: torch.optim.Optimizer) -> float | None:
        """Unscale, clip, step, zero. Returns the grad norm when clipping is on."""
        norm = None
        if self.clip_grad is not None:
            self.scaler.unscale_(optimizer)
            norm = float(torch.nn.utils.clip_grad_norm_(self.raw.parameters(), self.clip_grad))
        self.scaler.step(optimizer)
        self.scaler.update()
        optimizer.zero_grad(set_to_none=True)
        return norm

    @property
    def raw(self) -> TaskModule:
        """The module underneath any DDP wrapper."""
        m = self.module
        return m.module if isinstance(m, DistributedDataParallel) else m

    def _emit(self, hook: str, **kw: Any) -> None:
        for cb in self.callbacks:
            if fn := getattr(cb, hook, None):
                fn(trainer=self, module=self.raw, state=self.state, **kw)

    def _log(self, metrics: dict[str, float], step: int) -> None:
        if self.logger is not None and is_rank_zero():
            self.logger.log_scalars(metrics, step)

    def is_better(self, value: float) -> bool:
        best = self.state.best_metric
        if best is None:
            return True
        return value < best if self.monitor_mode == "min" else value > best

    # -- fit -----------------------------------------------------------------------

    def fit(
        self, module: TaskModule, datamodule: DataModule, resume: str | Path | None = None
    ) -> float | None:
        self.module, self.datamodule = module, datamodule
        module.trainer = self

        datamodule.setup("fit")
        module.to(device=self.device, dtype=self.param_dtype)

        spec = module.configure_optimizers()
        if not isinstance(spec, OptimSpec):
            raise TypeError(
                f"configure_optimizers must return OptimSpec, got {type(spec).__name__}"
            )
        self.optimizers, self.schedulers = spec.optimizers, spec.schedulers

        # Resume happens HERE, not in the caller: configure_optimizers() above builds
        # fresh optimizers, so anything loaded before fit() would be silently thrown
        # away and the run would continue with a cold optimizer -- no error, just
        # different training.
        if resume:
            from dlt.core.checkpoint import load_checkpoint

            log.info("resuming from %s", resume)
            load_checkpoint(
                resume,
                module,
                optimizers=self.optimizers,
                schedulers=self.schedulers,
                state=self.state,
                datamodule=datamodule,
            )
            log.info("resumed at step %d (epoch %d)", self.state.global_step, self.state.epoch)

        if self.compile:
            self.module = torch.compile(module, mode=self.compile_mode)
        if get_world_size() > 1:
            self.module = DistributedDataParallel(
                self.module, device_ids=[self.device.index] if self.device.type == "cuda" else None
            )

        loader = datamodule.train_dataloader()
        stream = _BatchStream(loader)
        eff = effective_batch_size(getattr(loader, "batch_size", None), self.grad_accum)
        log.info(
            "device=%s dtype=%s amp=%s | grad_accum=%d world=%d | effective batch=%s",
            self.device,
            self.param_dtype,
            self.amp_dtype,
            self.grad_accum,
            get_world_size(),
            eff if eff else "?",
        )

        self._emit("on_fit_start")
        acc, t0 = MetricAccumulator(), time.perf_counter()

        try:
            while self.state.global_step < self.max_steps and not self.state.should_stop:
                self.train_step(stream, acc)
                step = self.state.global_step

                if stream.epoch_just_ended:
                    self.state.epoch = stream.epoch
                    self._step_schedulers("epoch")
                    self._emit("on_epoch_end")
                    if self.max_epochs is not None and stream.epoch >= self.max_epochs:
                        break

                if self.log_every and step % self.log_every == 0:
                    metrics = acc.compute()
                    metrics["perf/steps_per_sec"] = self.log_every / max(
                        time.perf_counter() - t0, 1e-9
                    )
                    self.state.metrics.update(metrics)
                    self._log(metrics, step)
                    acc.reset()
                    t0 = time.perf_counter()

                if self.val_every and step % self.val_every == 0:
                    self._run_eval("validation_step", "val", step)
                if self.rollout_every and step % self.rollout_every == 0:
                    self._run_eval("rollout_step", "rollout", step)

            # Always end with a comparable metric, even if max_steps was not a
            # multiple of val_every -- but skip it if the loop just evaluated here,
            # which would duplicate both the work and the logged row.
            if self._last_val_step != self.state.global_step:
                self._run_eval("validation_step", "val", self.state.global_step)
        except BaseException:
            self._emit("on_exception")
            raise
        finally:
            self._emit("on_fit_end")

        return self.state.metrics.get(self.monitor)

    # -- one optimizer step (grad_accum micro-batches) ------------------------------

    def train_step(self, stream: _BatchStream, acc: MetricAccumulator) -> None:
        """Override this to change the update rule; everything else stays."""
        self.raw.train()
        manual = self.raw.manual_optimization

        for micro in range(self.grad_accum):
            batch = move_to_device(stream.next(), self.device, self.param_dtype)
            last = micro == self.grad_accum - 1

            # Skip the redundant all-reduce on every micro-step but the last.
            sync = (
                self.module.no_sync()
                if isinstance(self.module, DistributedDataParallel) and not last
                else contextlib.nullcontext()
            )
            with sync, self.autocast():
                out = self.raw.training_step(batch, self.state)

            if not manual:
                self.backward(out["loss"] / self.grad_accum)

            weight = float(out.get("batch_size", infer_batch_size(batch)))
            acc.update({f"train/{k}": v for k, v in out.items() if k != "batch_size"}, weight)
            self.state.samples_seen += int(weight)

        if not manual:
            for opt in self.optimizers:
                self.clip_and_step(opt)

        # global_step counts OPTIMIZER steps, not micro-batches. Counting micro-batches
        # would put every schedule and every logged x-axis off by grad_accum.
        self.state.global_step += 1
        self._step_schedulers("step")
        self._emit("on_train_batch_end")

    def _step_schedulers(self, interval: str) -> None:
        for s in self.schedulers:
            if s.scheduler is None or s.interval != interval:
                continue
            if self.state.global_step % max(s.frequency, 1):
                continue
            if s.monitor is not None:  # ReduceLROnPlateau and friends
                if (v := self.state.metrics.get(s.monitor)) is not None:
                    s.scheduler.step(v)
            else:
                s.scheduler.step()

    # -- evaluation -----------------------------------------------------------------

    def _grad_ctx(self) -> Any:
        """Models with physics losses cannot compute residuals inside no_grad.
        torch.inference_mode() is never used -- it is stricter and taints tensors
        against re-entering autograd later."""
        if self.raw.eval_requires_grad:
            return contextlib.nullcontext()
        return torch.no_grad()

    def _run_eval(self, method: str, prefix: str, step: int) -> None:
        loaders = self.datamodule.val_dataloader() if self.datamodule else None
        if loaders is None:
            return
        named = loaders.items() if isinstance(loaders, dict) else [("", loaders)]

        self.raw.eval()
        out: dict[str, float] = {}
        for name, loader in named:
            acc = MetricAccumulator()
            with self._grad_ctx():
                for batch in loader:
                    batch = move_to_device(batch, self.device, self.param_dtype)
                    res = getattr(self.raw, method)(batch, self.state)
                    if not res:
                        continue
                    weight = float(res.get("batch_size", infer_batch_size(batch)))
                    acc.update({k: v for k, v in res.items() if k != "batch_size"}, weight)
            ns = f"{prefix}/{name}/" if name else f"{prefix}/"
            out.update({ns + k: v for k, v in acc.compute().items()})
        self.raw.train()

        if prefix == "val":
            self._last_val_step = step
        if not out:
            return
        self.state.metrics.update(out)
        self._log(out, step)
        if prefix == "val":
            self._emit("on_val_end", metrics=out)
