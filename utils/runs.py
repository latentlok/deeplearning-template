#!/usr/bin/env python3
"""Query your runs.

The index is DERIVED, not maintained. Hydra already writes .hydra/config.yaml and
.hydra/overrides.yaml into every run dir, and each run writes its own run_meta.json --
so there is no shared append-and-rewrite file to corrupt under concurrent multirun.
Scanning a few thousand runs takes about a second, which is a good trade for deleting
a race condition.

    python utils/runs.py                          # everything, newest first
    python utils/runs.py --exp pinn --sort val/loss
    python utils/runs.py --where optimizer.lr=3e-4
    python utils/runs.py --status failed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def load_runs(root: Path) -> list[dict[str, Any]]:
    runs = []
    for meta_path in sorted(root.rglob("run_meta.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        d = meta_path.parent
        meta["path"] = str(d.relative_to(root))
        meta["overrides"] = _overrides(d / ".hydra" / "overrides.yaml")
        runs.append(meta)
    return runs


def _overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        items = yaml.safe_load(path.read_text()) or []
    except (OSError, yaml.YAMLError):
        return {}
    out = {}
    for item in items:
        key, _, val = str(item).partition("=")
        out[key.lstrip("+~")] = val
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("root", nargs="?", default="outputs", type=Path)
    p.add_argument("--exp", help="filter by experiment name")
    p.add_argument("--status", choices=["running", "finished", "failed"])
    p.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="filter by a config override, e.g. --where optimizer.lr=3e-4",
    )
    p.add_argument("--sort", metavar="METRIC", help="sort ascending by a metric, e.g. val/loss")
    p.add_argument("--limit", type=int, default=40)
    p.add_argument("--json", action="store_true", help="emit raw JSON instead of a table")
    args = p.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"no such directory: {args.root}")

    runs = load_runs(args.root)
    if args.exp:
        runs = [r for r in runs if r.get("exp") == args.exp]
    if args.status:
        runs = [r for r in runs if r.get("status") == args.status]
    for cond in args.where:
        key, _, val = cond.partition("=")
        runs = [r for r in runs if _matches(r["overrides"].get(key), val)]

    if args.sort:
        runs = [r for r in runs if args.sort in (r.get("metrics") or {})]
        runs.sort(key=lambda r: r["metrics"][args.sort])
    else:
        runs.sort(key=lambda r: r.get("started", ""), reverse=True)
    runs = runs[: args.limit]

    if args.json:
        print(json.dumps(runs, indent=2))
        return
    if not runs:
        print("no runs matched")
        return

    metric_keys = sorted({k for r in runs for k in (r.get("metrics") or {})})
    if args.sort:
        metric_keys = [args.sort] + [k for k in metric_keys if k != args.sort]
    metric_keys = metric_keys[:3]

    head = ["run", "status", "git"] + metric_keys
    rows = [
        [
            r["path"],
            r.get("status", "?") + ("*" if r.get("git_dirty") else ""),
            (r.get("git_sha") or "-")[:7],
            *[_fmt((r.get("metrics") or {}).get(k)) for k in metric_keys],
        ]
        for r in runs
    ]
    widths = [max(len(str(x)) for x in col) for col in zip(head, *rows, strict=True)]
    line = lambda row: "  ".join(str(c).ljust(w) for c, w in zip(row, widths, strict=True))  # noqa: E731
    print(line(head))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(line(row))
    print(f"\n{len(runs)} run(s). '*' marks a dirty working tree.")


def _matches(have: str | None, want: str) -> bool:
    """Literal match first, then numeric.

    Overrides are stored as the strings you typed, so a literal-only comparison makes
    `--where model.lr=0.005` silently return nothing for a run launched as
    `model.lr=5e-3` -- the same number, spelled differently.
    """
    if have is None:
        return False
    if have == want:
        return True
    try:
        return float(have) == float(want)
    except ValueError:
        return False


def _fmt(v: Any) -> str:
    return "-" if v is None else (f"{v:.5g}" if isinstance(v, int | float) else str(v))


if __name__ == "__main__":
    main()
