"""Small, boring helpers. Seeding, device movement, precision, schedules, provenance."""

from __future__ import annotations

import hashlib
import logging
import math
import os
import random
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Precision. dtype and amp are ORTHOGONAL axes, not one "precision" field.
#
# dtype is what the model *lives in*; amp is a runtime autocast wrapper on top of
# fp32. They are mutually exclusive, so a single field would imply you can pick both.
# float64 is not exotic: models with second derivatives routinely diverge in fp32,
# and create_graph=True compounds it.
# ---------------------------------------------------------------------------

DTYPES: dict[str, torch.dtype] = {"float32": torch.float32, "float64": torch.float64}
AMP_DTYPES: dict[str, torch.dtype | None] = {
    "none": None,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def resolve_precision(dtype: str, amp: str) -> tuple[torch.dtype, torch.dtype | None]:
    """Validate the pair up front, so a conflict is a config error at startup rather
    than a confusing failure deep in a backward pass."""
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    if amp not in AMP_DTYPES:
        raise ValueError(f"amp must be one of {sorted(AMP_DTYPES)}, got {amp!r}")
    if amp != "none" and dtype != "float32":
        raise ValueError(
            f"amp={amp!r} requires dtype='float32' (autocast wraps fp32); got dtype={dtype!r}. "
            "For fp64 runs use amp='none'."
        )
    return DTYPES[dtype], AMP_DTYPES[amp]


def set_default_dtype(dtype: torch.dtype) -> None:
    """Must run BEFORE the model is constructed. Set it after and the model is fp32
    while the data is fp64, which surfaces as a confusing matmul error rather than a
    config error.

    Called unconditionally every run: Hydra's basic launcher runs multirun jobs
    sequentially in ONE process, so a dtype set by job 1 would otherwise leak into job 2.
    """
    torch.set_default_dtype(dtype)


def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Also unconditional, for the same same-process-multirun reason as dtype."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Device movement. Duck-typed on purpose: the Trainer never inspects a batch, so
# dicts, tuples, namedtuples, graph objects and anything else with .to() all work
# without the framework knowing what they are.
# ---------------------------------------------------------------------------


def move_to_device(obj: Any, device: torch.device, dtype: torch.dtype | None = None) -> Any:
    if torch.is_tensor(obj):
        if dtype is not None and obj.is_floating_point():
            return obj.to(device=device, dtype=dtype)
        return obj.to(device=device)
    if isinstance(obj, Mapping):
        return {k: move_to_device(v, device, dtype) for k, v in obj.items()}
    if isinstance(obj, tuple) and hasattr(obj, "_fields"):  # namedtuple
        return type(obj)(*(move_to_device(v, device, dtype) for v in obj))
    if isinstance(obj, list | tuple):
        return type(obj)(move_to_device(v, device, dtype) for v in obj)
    if hasattr(obj, "to"):  # graph batches, custom containers
        return obj.to(device)
    return obj


def cast_module(module: Any, device: torch.device, dtype: torch.dtype) -> Any:
    """Move and cast a module without destroying complex weights.

    nn.Module.to(dtype=float32) casts complex parameters as well -- torch's _apply
    converts anything is_floating_point() OR is_complex() -- so a complex64 spectral
    weight silently becomes float32 and loses its imaginary part, with nothing but a
    ComplexWarning. Complex follows the real dtype instead: float32 -> complex64,
    float64 -> complex128.
    """
    complex_dtype = {torch.float32: torch.complex64, torch.float64: torch.complex128}.get(dtype)

    def convert(t: torch.Tensor) -> torch.Tensor:
        if t.is_complex():
            return t.to(device=device, dtype=complex_dtype)
        if t.is_floating_point():
            return t.to(device=device, dtype=dtype)
        return t.to(device=device)  # ints, bools, masks: move, never cast

    return module._apply(convert)


def infer_batch_size(batch: Any, default: int = 1) -> int:
    """Best-effort size of a batch, used to weight metric averages.

    Mean-of-means is wrong for variable-length sequences and variable-size batches, so
    metrics are weighted by this. Note it reports dim 0 -- for two-level sampling that
    is *items*, not points. Return "batch_size" from your step to override.
    """
    if torch.is_tensor(batch):
        return int(batch.shape[0]) if batch.ndim else default
    if isinstance(batch, Mapping):
        for v in batch.values():
            if n := infer_batch_size(v, 0):
                return n
    elif isinstance(batch, list | tuple):
        for v in batch:
            if n := infer_batch_size(v, 0):
                return n
    elif hasattr(batch, "num_graphs"):
        return int(batch.num_graphs)
    return default


class MetricAccumulator:
    """Batch-size-weighted running means. NaN/inf are dropped rather than poisoning."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._counts: dict[str, float] = {}

    def update(self, metrics: Mapping[str, Any], weight: float = 1.0) -> None:
        for k, v in metrics.items():
            try:
                val = float(v.detach().item()) if torch.is_tensor(v) else float(v)
            except (ValueError, TypeError, RuntimeError):
                continue  # non-scalar extras are not metrics
            if math.isnan(val) or math.isinf(val):
                continue
            self._sums[k] = self._sums.get(k, 0.0) + val * weight
            self._counts[k] = self._counts.get(k, 0.0) + weight

    def compute(self) -> dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums if self._counts[k]}

    def reset(self) -> None:
        self._sums.clear()
        self._counts.clear()


# ---------------------------------------------------------------------------
# ScheduledValue: one object for teacher-forcing ratios, loss-weight ramps, KL
# annealing, EMA decay warmup, curriculum difficulty.
#
# Inline `eps = max(0.0, 1 - step/20000)` would be one clear line -- but it is not
# sweepable, and you will want `-m teacher_forcing.over_steps=5000,20000,50000`.
# That is the entire argument for making it an object.
# ---------------------------------------------------------------------------

SCHEDULES = ("constant", "linear", "cosine", "exponential", "inverse_sigmoid")


class ScheduledValue:
    def __init__(
        self,
        start: float,
        end: float = 0.0,
        over_steps: int = 1,
        schedule: str = "linear",
        delay_steps: int = 0,
    ) -> None:
        if schedule not in SCHEDULES:
            raise ValueError(f"schedule must be one of {SCHEDULES}, got {schedule!r}")
        if schedule == "exponential" and (start <= 0 or end <= 0):
            raise ValueError("exponential schedule requires start > 0 and end > 0")
        self.start, self.end = float(start), float(end)
        self.over_steps, self.delay_steps = int(over_steps), int(delay_steps)
        self.schedule = schedule

    def __call__(self, step: int) -> float:
        if self.schedule == "constant":
            return self.start
        p = min(max((step - self.delay_steps) / max(self.over_steps, 1), 0.0), 1.0)
        a, b = self.start, self.end
        if self.schedule == "linear":
            return a + (b - a) * p
        if self.schedule == "cosine":
            return b + (a - b) * 0.5 * (1.0 + math.cos(math.pi * p))
        if self.schedule == "exponential":
            return a * (b / a) ** p
        # inverse_sigmoid, rescaled so it actually REACHES both endpoints. A raw
        # sigmoid only asymptotes: annealing a teacher-forcing ratio to 0 would stall
        # near 0.002 and you would never reach free-running.
        s = lambda q: 1.0 / (1.0 + math.exp(12.0 * (q - 0.5)))  # noqa: E731
        lo, hi = s(1.0), s(0.0)
        return b + (a - b) * (s(p) - lo) / (hi - lo)

    def __repr__(self) -> str:
        return f"ScheduledValue({self.schedule}, {self.start}->{self.end} over {self.over_steps})"


# ---------------------------------------------------------------------------
# Distributed. Rank-aware from the start: retrofitting means auditing every log and
# save call.
# ---------------------------------------------------------------------------


def get_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_rank_zero() -> bool:
    return get_rank() == 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def resolve_device(spec: str = "auto") -> torch.device:
    """ "auto" -> cuda:LOCAL_RANK when a GPU is present, else cpu.

    The rank index is not cosmetic: torchrun starts one process per GPU and a bare
    "cuda" lands every one of them on cuda:0. Architecture compatibility is
    deliberately NOT probed -- on a mismatched box pass trainer.device=cpu.
    """
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda", get_local_rank() % torch.cuda.device_count())
    return torch.device("cpu")


def init_distributed(device: torch.device) -> bool:
    """Join the process group. Returns True if this call created it.

    torchrun sets RANK/WORLD_SIZE but never calls init_process_group for you, and
    DistributedDataParallel refuses to construct without it -- so a multi-rank run
    dies at the wrap with "Default process group has not been initialized".
    """
    if get_world_size() <= 1 or torch.distributed.is_initialized():
        return False
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.distributed.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    return True


def effective_batch_size(batch_size: int | None, grad_accum: int) -> int | None:
    """batch_size x grad_accum x world_size. Logged at startup because this is the
    number people get wrong and then cannot reproduce."""
    if not batch_size:
        return None
    return batch_size * grad_accum * get_world_size()


# ---------------------------------------------------------------------------
# Provenance and run naming.
# ---------------------------------------------------------------------------


def git_info(cwd: Path | None = None) -> dict[str, Any]:
    """Returns {} outside a git repo rather than raising."""

    def _run(*args: str) -> str | None:
        try:
            r = subprocess.run(
                args, cwd=cwd, capture_output=True, text=True, timeout=5, check=False
            )
            return r.stdout.strip() if r.returncode == 0 else None
        except (OSError, subprocess.SubprocessError):
            return None

    sha = _run("git", "rev-parse", "HEAD")
    if sha is None:
        return {}
    return {
        "git_sha": sha,
        "git_dirty": bool(_run("git", "status", "--porcelain")),
        "git_branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
    }


def _canonical(obj: Any) -> str:
    if isinstance(obj, Mapping):
        return "{" + ",".join(f"{k}:{_canonical(obj[k])}" for k in sorted(map(str, obj))) + "}"
    if isinstance(obj, list | tuple):
        return "[" + ",".join(_canonical(v) for v in obj) + "]"
    return repr(obj)


def config_hash(cfg: DictConfig | dict, length: int = 6) -> str:
    """Hash of the fully-resolved config. This is the authoritative one.

    The directory name uses the cheaper override hash below, which must be available
    at hydra.run.dir interpolation time -- before the resolved config exists. That one
    can collide across genuinely different configs (edit a YAML in place and the
    override string is unchanged), so run metadata records this instead.
    """
    container = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    return hashlib.blake2b(_canonical(container).encode(), digest_size=length // 2 + 1).hexdigest()[
        :length
    ]


def run_hash(overrides: str = "", length: int = 6) -> str:
    """Short hash of the sorted override set, for the run directory name.

    Derived from hydra.job.override_dirname, passed in as an argument rather than read
    from HydraConfig inside the resolver -- the argument is reliably available at
    hydra.run.dir interpolation time; HydraConfig is not.
    """
    key = "|".join(sorted(p for p in str(overrides).split(",") if p))
    return hashlib.blake2b(key.encode(), digest_size=length // 2 + 1).hexdigest()[:length]


def ckpt_run_dir(ckpt: Any) -> str:
    """The run directory that produced a checkpoint.

        outputs/pinn/2026-08-03_14-22-05_a3f9c2/ckpt/best  ->
        outputs/pinn/2026-08-03_14-22-05_a3f9c2

    Used by eval.yaml so an evaluation writes INTO the run it evaluated, next to the
    config and the weights that produced it. A separate outputs/eval/<timestamp>/ tree
    is orphaned the moment you have two runs of the same experiment: the numbers are
    real but nothing on disk says which weights made them.

    Walks up to the nearest ancestor named `ckpt`; anything else (a hand-placed
    checkpoint, a downloaded one) falls back to the checkpoint's own parent, which is
    still beside the weights rather than in a global bucket.
    """
    if not ckpt or str(ckpt) == "???":
        raise ValueError(
            "eval needs a checkpoint: eval.py ckpt=outputs/<exp>/<run>/ckpt/best "
            "(the output directory is derived from it)"
        )
    p = Path(str(ckpt)).expanduser()
    for parent in p.parents:
        if parent.name == "ckpt":
            return str(parent.parent)
    return str(p.parent)


def register_resolvers() -> None:
    """Must run before @hydra.main composes.

    use_cache=True is REQUIRED for run_hash: without it each interpolation
    re-evaluates and one run scatters across several directories. ckpt_run_dir is a
    pure function of its argument, so it needs no cache. replace=True keeps
    same-process multirun and repeated test imports from raising on re-registration.
    """
    OmegaConf.register_new_resolver("run_hash", run_hash, use_cache=True, replace=True)
    OmegaConf.register_new_resolver("ckpt_run_dir", ckpt_run_dir, replace=True)
