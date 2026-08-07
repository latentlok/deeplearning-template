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

from engine.base import DataModule, OptimSpec, TaskModule, TrainState
from engine.utils import (
    MetricAccumulator,
    cast_module,
    effective_batch_size,
    get_world_size,
    infer_batch_size,
    init_distributed,
    is_rank_zero,
    move_to_device,
    resolve_device,
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


class _StepWrapper(torch.nn.Module):
    """Makes `training_step` the forward that DDP and torch.compile actually see.

    Both only act on graphs built inside their own forward. Calling
    module.training_step directly walks straight past them: DDP's reducer never runs
    prepare_for_backward so nothing is all-reduced and the ranks silently diverge,
    and torch.compile never traces a single frame. Neither raises.

    Used only when compile or world_size > 1 is on; single-device training keeps the
    direct call and this class never appears.
    """

    def __init__(self, task: TaskModule) -> None:
        super().__init__()
        self._task = task

    def forward(self, batch: Any, state: TrainState) -> dict[str, torch.Tensor]:
        return self._task.training_step(batch, state)


def _warn_if_unsharded(loaders: Any) -> None:
    """Sharding is the DataModule's job. Staying silent about it is not.

    With no DistributedSampler every rank draws the SAME batches, so N ranks all-reduce
    N identical gradients: N times the compute for one device's worth of data, no error
    anywhere, and a loss curve that looks perfectly healthy.
    """
    if get_world_size() <= 1:
        return
    from torch.utils.data import IterableDataset
    from torch.utils.data.distributed import DistributedSampler

    for lo in loaders.values() if isinstance(loaders, dict) else [loaders]:
        ds = getattr(lo, "dataset", None)
        if isinstance(ds, IterableDataset):  # streaming shards inside the dataset
            continue
        if not isinstance(getattr(lo, "sampler", None), DistributedSampler):
            log.warning(
                "world_size=%d but the train loader has no DistributedSampler -- every "
                "rank will draw the same batches. Build one in your DataModule.",
                get_world_size(),
            )
            return


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

        self.device = resolve_device(device)
        # GradScaler is only meaningful for fp16; bf16 has fp32's exponent range.
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=(amp_dtype is torch.float16))
        self._owns_pg = False
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
        """The TaskModule underneath any DDP / compile / step wrapper.

        `_task` resolves through torch.compile's OptimizedModule too, which forwards
        unknown attributes to the module it wrapped.
        """
        m = self.module
        if isinstance(m, DistributedDataParallel):
            m = m.module
        return getattr(m, "_task", m)

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

        # Before anything touches the device: torchrun exports the env vars but never
        # creates the process group, and DDP will not construct without one.
        self._owns_pg = init_distributed(self.device)

        datamodule.setup("fit")
        # The only data -> model handoff, and it runs BEFORE the cast and before
        # resume: statistics are written into buffers, so the cast must see their final
        # dtype and a checkpoint's stored values must win over whatever the data just
        # reported.
        module.on_data_ready(datamodule)
        cast_module(module, self.device, self.param_dtype)

        # Wrap BEFORE building optimizers. A strategy that replaces parameter objects
        # rather than wrapping them -- FSDP2's fully_shard swaps every Parameter for a
        # sharded DTensor -- would otherwise leave the optimizer holding the pre-wrap
        # tensors, which never receive a gradient: the run trains cleanly, logs a loss,
        # and updates nothing. DDP and compile keep the same objects either way.
        self.module = self.wrap(module)

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
            from engine.checkpoint import load_checkpoint

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

        loader = datamodule.train_dataloader()
        _warn_if_unsharded(loader)
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
            if self._owns_pg:
                torch.distributed.destroy_process_group()
                self._owns_pg = False

        return self.state.metrics.get(self.monitor)

    def wrap(self, module: TaskModule) -> Any:
        """Compile and distributed wrapping, in one overridable place.

        Override to swap the strategy without reimplementing fit() -- FSDP2 is
        `fully_shard(module); return module`, since it shards in place and needs no
        wrapper object. Whatever this returns is what the loop calls forward on, and
        `raw` must still resolve to the TaskModule.

        Both wrappers here must OWN the forward or they silently do nothing; see
        _StepWrapper. Single-device eager training returns the module untouched.
        """
        wrapped: Any = module
        if self.compile or get_world_size() > 1:
            wrapped = _StepWrapper(module)
        if self.compile:
            wrapped = torch.compile(wrapped, mode=self.compile_mode)
        if get_world_size() > 1:
            wrapped = DistributedDataParallel(
                wrapped, device_ids=[self.device.index] if self.device.type == "cuda" else None
            )
        return wrapped

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
                # Through the wrapper when there is one, never around it.
                out = (
                    self.module(batch, self.state)
                    if self.module is not self.raw
                    else self.raw.training_step(batch, self.state)
                )

            if not manual:
                self.backward(out["loss"] / self.grad_accum)

            weight = float(out.get("batch_size", infer_batch_size(batch)))
            acc.update({f"train/{k}": v for k, v in out.items() if k != "batch_size"}, weight)
            self.state.samples_seen += int(weight)

        # Gradients are alive HERE and gone immediately after, because clip_and_step
        # ends with zero_grad(set_to_none=True). Anything that inspects gradients must
        # run on this hook -- on_train_batch_end is too late and sees None.
        self._emit("on_before_optimizer_step")

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
