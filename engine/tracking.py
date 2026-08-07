"""Observability -- the actual product of this template.

Three channels, all rank-zero-only, all independently toggleable:

  1. TensorBoard  -> <run_dir>/tb/
  2. Python logging -> <run_dir>/train.log   (configured by Hydra job_logging)
  3. JSONL        -> <run_dir>/metrics.jsonl

Plus a single-writer <run_dir>/run_meta.json, and <run_dir>/artifacts/ for anything
that is neither a scalar nor a figure.

Hydra already writes .hydra/{config,overrides,hydra}.yaml into every run dir, so the
resolved config is NOT snapshotted here -- that would be a straight duplicate.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from engine.utils import git_info, is_rank_zero

log = logging.getLogger(__name__)


def artifacts_dir(run_dir: str | Path, name: str | None = None) -> Path:
    """Return (and create) the artifacts directory for a run.

    Deliberately usable *outside* Hydra, because manual inference happens in a
    notebook, not in a @hydra.main job:

        from engine.tracking import artifacts_dir
        d = artifacts_dir("outputs/pinn/2026-08-03_14-22-05_a3f9c2", "rollout_h1000")
        torch.save(traj, d / "traj.pt")

    Living inside the run dir means an output file can never drift from the config and
    weights that produced it. That is the whole feature -- a path, created on demand.
    """
    d = Path(run_dir) / "artifacts"
    if name:
        d = d / name
    d.mkdir(parents=True, exist_ok=True)
    return d


class Logger(ABC):
    """Everything is optional except log_scalars."""

    @abstractmethod
    def log_scalars(self, metrics: dict[str, float], step: int) -> None: ...

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float]) -> None: ...
    def log_figure(self, tag: str, figure: Any, step: int) -> None: ...
    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None: ...
    def log_text(self, tag: str, text: str, step: int = 0) -> None: ...
    def close(self) -> None: ...


class ConsoleLogger(Logger):
    """Plain Python logging. Formatting is Hydra's job_logging; this only decides
    what is worth a line."""

    def __init__(self, every: int = 1, max_keys: int = 6) -> None:
        self.every, self.max_keys = every, max_keys

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        if self.every and step % self.every:
            return
        keys = sorted(metrics, key=lambda k: (not k.endswith("loss"), k))[: self.max_keys]
        body = " ".join(f"{k}={metrics[k]:.4g}" for k in keys)
        log.info("step %-7d %s", step, body)


class JSONLLogger(Logger):
    """One JSON object per logging step. The headless workhorse: jq-able,
    pandas-loadable, and what aggregate_seeds.py reads."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        rec = {"step": step, **{k: _py(v) for k, v in metrics.items()}}
        self._fh.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


class TensorBoardLogger(Logger):
    """Writes to <run_dir>/tb/, so `tensorboard --logdir outputs` shows every run."""

    def __init__(self, log_dir: str | Path, flush_secs: int = 30) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir=str(self.log_dir), flush_secs=flush_secs)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        for k, v in metrics.items():
            self.writer.add_scalar(k, _py(v), step)

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        self.writer.add_figure(tag, figure, step, close=True)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        """Complex tensors are split rather than passed through.

        add_histogram does NOT reject complex -- it silently casts to real and drops
        the imaginary part with only a ComplexWarning. A silent half-truth in your
        diagnostics is worse than a crash.
        """
        v = values.detach()
        if v.is_complex():
            self.writer.add_histogram(f"{tag}/abs", v.abs().float().cpu(), step)
            self.writer.add_histogram(f"{tag}/real", v.real.float().cpu(), step)
            self.writer.add_histogram(f"{tag}/imag", v.imag.float().cpu(), step)
            return
        self.writer.add_histogram(tag, v.float().cpu(), step)  # float() also handles fp64

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        self.writer.add_text(tag, text, step)

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float]) -> None:
        """Populates the HPARAMS tab, which is what makes runs comparable.

        Two gotchas handled: TB silently drops runs whose hparams contain lists/dicts,
        so values are flattened to dotted keys and filtered to int/float/str/bool; and
        without run_name="." the results land in a nested subdirectory.
        """
        clean = {k: v for k, v in hparams.items() if isinstance(v, int | float | str | bool)}
        if not clean or not metrics:
            return
        self.writer.add_hparams(clean, {k: _py(v) for k, v in metrics.items()}, run_name=".")

    def close(self) -> None:
        self.writer.flush()
        self.writer.close()


class MultiLogger(Logger):
    """Fan-out. Rank-zero gating lives here, once, rather than in every call site."""

    def __init__(self, loggers: list[Logger] | None = None) -> None:
        self.loggers = [x for x in (loggers or []) if x is not None] if is_rank_zero() else []

    def _fan(self, method: str, *a: Any, **kw: Any) -> None:
        for lg in self.loggers:
            try:
                getattr(lg, method)(*a, **kw)
            except Exception:  # a broken logger must never kill a training run
                log.exception("logger %s.%s failed", type(lg).__name__, method)

    def log_scalars(self, metrics: dict[str, float], step: int) -> None:
        self._fan("log_scalars", metrics, step)

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float]) -> None:
        self._fan("log_hparams", hparams, metrics)

    def log_figure(self, tag: str, figure: Any, step: int) -> None:
        self._fan("log_figure", tag, figure, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int) -> None:
        self._fan("log_histogram", tag, values, step)

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        self._fan("log_text", tag, text, step)

    def close(self) -> None:
        self._fan("close")


# ---------------------------------------------------------------------------
# run_meta.json -- per-run, single writer, so no locking and no race. A shared
# append-and-rewrite index would corrupt under concurrent multirun; the queryable
# table is DERIVED by scripts/runs.py instead of maintained.
# ---------------------------------------------------------------------------


class RunMeta:
    def __init__(self, run_dir: str | Path, exp_name: str, config_hash: str = "") -> None:
        self.path = Path(run_dir) / "run_meta.json"
        self.data: dict[str, Any] = {
            "exp": exp_name,
            "run_dir": str(run_dir),
            "config_hash": config_hash,
            "started": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": "running",
            "hostname": os.uname().nodename,
            "world_size": int(os.environ.get("WORLD_SIZE", 1)),
            "torch": torch.__version__,
            "metrics": {},
            **git_info(),
        }
        self.write()

    def finish(self, status: str, metrics: dict[str, float] | None = None) -> None:
        self.data["status"] = status
        self.data["ended"] = datetime.now(UTC).isoformat(timespec="seconds")
        if metrics:
            self.data["metrics"] = {k: _py(v) for k, v in metrics.items()}
        self.write()

    def write(self) -> None:
        if not is_rank_zero():
            return
        # Atomic replace: a crash mid-write must not leave unparseable JSON behind.
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(self.data, fh, indent=2, default=str)
        os.replace(tmp, self.path)


def _py(v: Any) -> Any:
    if torch.is_tensor(v):
        return v.detach().float().item() if v.numel() == 1 else v.detach().float().tolist()
    return v


def flatten_config(cfg: Any, prefix: str = "") -> dict[str, Any]:
    """Nested config -> dotted scalar keys, for the TB HPARAMS tab."""
    from omegaconf import DictConfig, ListConfig, OmegaConf

    if isinstance(cfg, DictConfig | ListConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    out: dict[str, Any] = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            out.update(flatten_config(v, f"{prefix}{k}."))
    elif isinstance(cfg, int | float | str | bool) or cfg is None:
        out[prefix.rstrip(".")] = cfg
    return out
