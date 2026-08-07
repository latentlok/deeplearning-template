#!/usr/bin/env python3
"""Compute normalisation statistics over the TRAIN split, once, offline.

    python utils/stats.py                          # uses $DL_DATA
    python utils/stats.py --root /mnt/data/proj
    python utils/stats.py --root /mnt/data/proj --split train --out /tmp/stats.json

Writes <root>/stats.json:

    {"count": 51200, "shape": [3, 64, 64],
     "lower": [...], "upper": [...], "mean": [...], "std": [...],
     "y": {"lower": [...], ...}}

FolderData loads that file in setup(); the model copies the numbers it needs into
buffers in on_data_ready, so they end up inside the checkpoint. That chain is the
whole point:

    train split  ->  stats.json  ->  model buffers  ->  checkpoint  ->  inference

Statistics computed on the fly at training time break it in two places. They cost
minutes per run, and an evaluation that recomputes them from a different split -- or
from data that has grown since -- silently normalises against different numbers than
training used. The predictions stay plausible, which is what makes it expensive.

Everything reduces over all axes except axis 1 (the channel / feature axis), so
(N, D) gives D numbers and (N, C, H, W) gives C. Pass --axis to change it, or
--axis -1 for a single scalar per array. The scan is chunked, so the array is never
fully resident.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.loader import _find, _open  # noqa: E402


def compute_stats(array: Any, axis: int = 1, chunk: int = 4096) -> dict[str, Any]:
    """Streaming min / max / mean / std, reduced over every axis but `axis`.

    Variance comes from sum and sum-of-squares in float64. The naive fp32 version of
    this is a classic source of a slightly-wrong std (and, on large N, a negative one).
    """
    n = len(array)
    if n == 0:
        raise ValueError("empty array")

    sample = np.asarray(array[0])
    keep = None if axis < 0 or sample.ndim == 0 else axis - 1  # sample has no batch dim
    if keep is not None and keep >= sample.ndim:
        raise ValueError(f"--axis {axis} is out of range for samples of shape {sample.shape}")

    shape = () if keep is None else (sample.shape[keep],)
    count = 0
    total = np.zeros(shape, dtype=np.float64)
    total_sq = np.zeros(shape, dtype=np.float64)
    lower = np.full(shape, np.inf, dtype=np.float64)
    upper = np.full(shape, -np.inf, dtype=np.float64)

    for start in range(0, n, chunk):
        block = np.asarray(array[start : start + chunk], dtype=np.float64)
        if keep is None:
            axes: tuple[int, ...] = tuple(range(block.ndim))
        else:
            axes = tuple(i for i in range(block.ndim) if i != keep + 1)
        count += int(np.prod([block.shape[i] for i in axes]))
        total += block.sum(axis=axes)
        total_sq += (block**2).sum(axis=axes)
        lower = np.minimum(lower, block.min(axis=axes))
        upper = np.maximum(upper, block.max(axis=axes))

    mean = total / count
    var = np.maximum(total_sq / count - mean**2, 0.0)  # clamp fp noise, never sqrt(-0)
    return {
        "count": int(n),
        "shape": list(sample.shape),
        "lower": lower.tolist(),
        "upper": upper.tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(var).tolist(),
    }


def main() -> None:
    import os

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=os.environ.get("DL_DATA", "data"))
    p.add_argument("--split", default="train", help="statistics come from TRAIN only")
    p.add_argument("--inputs", default="x")
    p.add_argument("--targets", default="y")
    p.add_argument("--axis", type=int, default=1, help="axis to keep; -1 for a scalar")
    p.add_argument("--chunk", type=int, default=4096, help="samples read at a time")
    p.add_argument("--out", type=Path, default=None, help="default: <root>/stats.json")
    args = p.parse_args()

    root = Path(args.root).expanduser()
    split = root / args.split
    if not split.is_dir():
        raise SystemExit(f"no such split directory: {split}")

    stats = compute_stats(_open(_find(split, args.inputs)), args.axis, args.chunk)
    stats["split"] = args.split
    try:
        stats["y"] = compute_stats(_open(_find(split, args.targets)), args.axis, args.chunk)
    except FileNotFoundError:
        pass  # inputs-only datasets are fine

    out = args.out or root / "stats.json"
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {out}")
    print(f"  count {stats['count']}  shape {stats['shape']}")
    print(f"  lower {_short(stats['lower'])}")
    print(f"  upper {_short(stats['upper'])}")


def _short(v: Any, k: int = 6) -> str:
    if not isinstance(v, list):
        return f"{v:.4g}"
    head = ", ".join(f"{x:.4g}" for x in v[:k])
    return f"[{head}{', ...' if len(v) > k else ''}]"


if __name__ == "__main__":
    main()
