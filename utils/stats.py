#!/usr/bin/env python3
"""Compute normalisation statistics over the TRAIN store, once, offline.

    python utils/stats.py                            # uses $DL_DATA
    python utils/stats.py --root /mnt/data/proj
    python utils/stats.py --root /mnt/data/proj --store train.zarr --out /tmp/stats.json

Writes <root>/stats.json, keyed BY VARIABLE NAME:

    {"feature_1": {"count": 4096, "mean": 0.01, "std": 0.99, "min": -3.7, "max": 3.9},
     "feature_2": {...},
     "feature_3": {...}}

Keyed by name, not by column position, so reordering or subsetting `data.inputs`
cannot silently pair a variable with another variable's mean. dataset/loader.py
assembles these into x_mean / x_std / y_mean / y_std in config order.

The chain this exists to serve:

    train store  ->  stats.json  ->  model buffers  ->  checkpoint  ->  inference

Statistics computed on the fly at training time break it in two places. They cost
minutes per run, and an evaluation that recomputes them from a different split -- or
from data that has grown since -- silently normalises against different numbers than
training used. The predictions stay plausible, which is what makes it expensive.

A (T,) variable gives one number per statistic; a (T, D) variable gives D of them,
reduced over rows. The scan is chunked, so the array is never fully resident.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.loader import open_group  # noqa: E402


def compute_stats(array: Any, chunk: int = 4096) -> dict[str, Any]:
    """Streaming count / mean / std / min / max, reduced over rows (axis 0).

    Variance comes from sum and sum-of-squares in float64. The naive fp32 version is a
    classic source of a slightly-wrong std, and on large N a negative one.
    """
    n = array.shape[0]  # zarr Arrays have no len()
    if n == 0:
        raise ValueError("empty array")

    width = np.atleast_1d(np.asarray(array[0])).shape[0]
    total = np.zeros(width, dtype=np.float64)
    total_sq = np.zeros(width, dtype=np.float64)
    lower = np.full(width, np.inf, dtype=np.float64)
    upper = np.full(width, -np.inf, dtype=np.float64)

    for start in range(0, n, chunk):
        block = np.asarray(array[start : start + chunk], dtype=np.float64).reshape(-1, width)
        total += block.sum(axis=0)
        total_sq += (block**2).sum(axis=0)
        lower = np.minimum(lower, block.min(axis=0))
        upper = np.maximum(upper, block.max(axis=0))

    mean = total / n
    var = np.maximum(total_sq / n - mean**2, 0.0)  # clamp fp noise, never sqrt(-0)
    return {
        "count": int(n),
        "mean": _scalar_or_list(mean),
        "std": _scalar_or_list(np.sqrt(var)),
        "min": _scalar_or_list(lower),
        "max": _scalar_or_list(upper),
    }


def _scalar_or_list(v: np.ndarray) -> Any:
    """A (T,) variable reads back as one number, not a one-element list."""
    return float(v[0]) if v.shape[0] == 1 else v.tolist()


def main() -> None:
    import os

    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--root", type=Path, default=os.environ.get("DL_DATA", "data"))
    p.add_argument("--store", default="train.zarr", help="statistics come from TRAIN only")
    p.add_argument("--vars", default=None, help="comma-separated; default: every variable")
    p.add_argument("--chunk", type=int, default=4096, help="rows read at a time")
    p.add_argument("--out", type=Path, default=None, help="default: <root>/stats.json")
    args = p.parse_args()

    root = Path(args.root).expanduser()
    group = open_group(root / args.store)
    names = args.vars.split(",") if args.vars else sorted(group.array_keys())

    stats = {name: compute_stats(group[name], args.chunk) for name in names}

    out = args.out or root / "stats.json"
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"wrote {out}  ({args.store})")
    for name, s in stats.items():
        print(f"  {name:<16} mean {_short(s['mean'])}  std {_short(s['std'])}")


def _short(v: Any, k: int = 4) -> str:
    if not isinstance(v, list):
        return f"{v:.4g}"
    head = ", ".join(f"{x:.4g}" for x in v[:k])
    return f"[{head}{', ...' if len(v) > k else ''}]"


if __name__ == "__main__":
    main()
