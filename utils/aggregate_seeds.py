#!/usr/bin/env python3
"""Mean +- std across a seed sweep, because a single-seed number is not a result.

python train.py -m experiment=pinn seed=1,2,3,4,5
python utils/aggregate_seeds.py outputs/pinn/2026-08-03_14-22-05_sweep
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def collect(root: Path) -> dict[str, list[float]]:
    """Reads run_meta.json where present, falling back to the last metrics.jsonl row
    so that a crashed-but-partial run still contributes."""
    series: dict[str, list[float]] = {}
    for meta_path in sorted(root.rglob("run_meta.json")):
        metrics = json.loads(meta_path.read_text()).get("metrics") or {}
        if not metrics:
            jsonl = meta_path.parent / "metrics.jsonl"
            if jsonl.exists():
                rows = [json.loads(x) for x in jsonl.read_text().splitlines() if x.strip()]
                metrics = rows[-1] if rows else {}
        for k, v in metrics.items():
            if isinstance(v, int | float) and k != "step":
                series.setdefault(k, []).append(float(v))
    return series


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("sweep_dir", type=Path)
    p.add_argument("--json", action="store_true")
    args = p.parse_args()

    if not args.sweep_dir.is_dir():
        raise SystemExit(f"no such directory: {args.sweep_dir}")

    series = collect(args.sweep_dir)
    if not series:
        raise SystemExit(f"no run_meta.json / metrics.jsonl found under {args.sweep_dir}")

    out: dict[str, Any] = {}
    for k in sorted(series):
        vals = series[k]
        out[k] = {
            "n": len(vals),
            "mean": statistics.fmean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    if args.json:
        print(json.dumps(out, indent=2))
        return

    width = max(len(k) for k in out)
    print(f"{'metric'.ljust(width)}  {'n':>3}  {'mean':>12}  {'std':>12}")
    print("-" * (width + 34))
    for k, s in out.items():
        print(f"{k.ljust(width)}  {s['n']:>3}  {s['mean']:>12.5g}  {s['std']:>12.5g}")


if __name__ == "__main__":
    main()
