"""Save, load, resume.

Weights go through a pluggable WeightFormat; everything else -- optimizers,
schedulers, TrainState, RNG, dataloader position -- goes to state.pt, which is
trusted-by-construction since you wrote it. Non-tensor state (tokenizers, vocabs,
scaler config) goes to extra/ via the module's save_extra hook.

The format is deliberately an OPEN extension point, not a closed enum. Core stays
general; a fork that needs dtype-specific handling (complex spectral weights, packed
quantised tensors, sharded writes) subclasses WeightFormat and points config at it:

    # configs/checkpoint/complex.yaml
    format:
      _target_: myproj.ckpt.ComplexSafetensors

Nothing in core changes.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from engine.base import DataModule, TaskModule, TrainState
from engine.utils import is_rank_zero

log = logging.getLogger(__name__)

STATE = "state.pt"
EXTRA = "extra"


class WeightFormat(ABC):
    """How model weights are written and read. Subclass to add your own."""

    filename: str

    @abstractmethod
    def save(self, module: torch.nn.Module, path: Path) -> None: ...

    @abstractmethod
    def load(self, module: torch.nn.Module, path: Path, map_location: Any = "cpu") -> None: ...


class SafetensorsFormat(WeightFormat):
    """Fast, zero-copy, no pickle execution risk.

    Supports a fixed dtype set -- complex128 in particular is not in it. Rather than
    silently converting, this says exactly what to change.
    """

    filename = "model.safetensors"

    def save(self, module: torch.nn.Module, path: Path) -> None:
        from safetensors.torch import save_model

        try:
            save_model(module, str(path))
        except Exception as e:
            raise RuntimeError(
                f"safetensors could not save these weights ({type(e).__name__}: {e}). "
                "Usually an unsupported dtype. Either set `checkpoint.format` to "
                "engine.checkpoint.TorchFormat (handles every torch dtype), or "
                "subclass WeightFormat for dtype-specific handling."
            ) from e

    def load(self, module: torch.nn.Module, path: Path, map_location: Any = "cpu") -> None:
        from safetensors.torch import load_model

        load_model(module, str(path))


class TorchFormat(WeightFormat):
    """torch.save/load. Handles every dtype; the no-questions escape hatch."""

    filename = "model.pt"

    def save(self, module: torch.nn.Module, path: Path) -> None:
        torch.save(module.state_dict(), path)

    def load(self, module: torch.nn.Module, path: Path, map_location: Any = "cpu") -> None:
        module.load_state_dict(torch.load(path, map_location=map_location))


_BUILTIN: dict[str, type[WeightFormat]] = {
    "safetensors": SafetensorsFormat,
    "torch": TorchFormat,
}


def resolve_format(fmt: str | WeightFormat) -> WeightFormat:
    if isinstance(fmt, WeightFormat):
        return fmt
    if fmt in _BUILTIN:
        return _BUILTIN[fmt]()
    raise ValueError(
        f"unknown checkpoint format {fmt!r}; use one of {sorted(_BUILTIN)} "
        "or pass a WeightFormat instance via _target_"
    )


def save_checkpoint(
    directory: str | Path,
    module: TaskModule,
    optimizers: list[Any] | None = None,
    schedulers: list[Any] | None = None,
    state: TrainState | None = None,
    datamodule: DataModule | None = None,
    fmt: str | WeightFormat = "safetensors",
    weights_only: bool = False,
) -> Path:
    """`weights_only=True` writes the weights and extra/ but no state.pt.

    That is the inference artifact: no optimizer moments, no RNG, no step counter --
    typically a third the size and impossible to accidentally resume from. extra/ is
    still written, because a tokenizer or scaler config is needed to *use* the model.
    """
    d = Path(directory)
    if not is_rank_zero():
        return d
    d.mkdir(parents=True, exist_ok=True)

    writer = resolve_format(fmt)
    writer.save(module, d / writer.filename)

    if weights_only:
        extra = d / EXTRA
        extra.mkdir(parents=True, exist_ok=True)
        module.save_extra(extra)
        return d

    torch.save(
        {
            "optimizers": [o.state_dict() for o in (optimizers or [])],
            "schedulers": [
                s.scheduler.state_dict()
                for s in (schedulers or [])
                if hasattr(s.scheduler, "state_dict")
            ],
            "state": state.state_dict() if state else {},
            "rng": {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            "data": datamodule.state_dict() if datamodule else {},
        },
        d / STATE,
    )

    extra = d / EXTRA
    extra.mkdir(parents=True, exist_ok=True)
    module.save_extra(extra)
    return d


def load_checkpoint(
    directory: str | Path,
    module: TaskModule,
    optimizers: list[Any] | None = None,
    schedulers: list[Any] | None = None,
    state: TrainState | None = None,
    datamodule: DataModule | None = None,
    map_location: Any = "cpu",
    weights_only: bool = False,
) -> None:
    """`weights_only=True` loads the model and skips optimizer/state -- the fine-tune
    or transfer case, which is genuinely different from an exact resume."""
    d = Path(directory)
    if not d.exists():
        raise FileNotFoundError(f"checkpoint directory not found: {d}")

    # extra/ first: a tokenizer or scaler may need to exist before weights land.
    if (d / EXTRA).exists():
        module.load_extra(d / EXTRA)

    for cls in _BUILTIN.values():
        if (d / cls.filename).exists():
            cls().load(module, d / cls.filename, map_location)
            break
    else:
        raise FileNotFoundError(
            f"no recognised weight file in {d} (looked for "
            f"{', '.join(c.filename for c in _BUILTIN.values())})"
        )

    if weights_only or not (d / STATE).exists():
        return

    blob = torch.load(d / STATE, map_location=map_location, weights_only=False)

    for opt, sd in zip(optimizers or [], blob.get("optimizers", []), strict=False):
        opt.load_state_dict(sd)
    for spec, sd in zip(schedulers or [], blob.get("schedulers", []), strict=False):
        if hasattr(spec.scheduler, "load_state_dict"):
            spec.scheduler.load_state_dict(sd)
    if state is not None and blob.get("state"):
        state.load_state_dict(blob["state"])

    if rng := blob.get("rng"):
        torch.set_rng_state(rng["cpu"])
        if torch.cuda.is_available() and rng.get("cuda"):
            torch.cuda.set_rng_state_all(rng["cuda"])

    if datamodule is not None:
        if data_sd := blob.get("data"):
            datamodule.load_state_dict(data_sd)
        else:
            # Say it out loud. Silently restarting a stream retrains the same prefix.
            log.warning(
                "resuming at step %s but the DataModule saved no state -- data position "
                "is best-effort and the loader restarts from the beginning. Implement "
                "DataModule.state_dict()/load_state_dict() to make it exact.",
                state.global_step if state else "?",
            )
