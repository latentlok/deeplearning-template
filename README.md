# dlt — a minimal, fork-first PyTorch training template

Train, track, log. Nothing else.

You cannot design one abstraction that covers PINNs, JEPA, neural operators, neural
ODEs and language models. Anything general enough to span them is worse than no
abstraction — so this doesn't try. It gives you **the loop, the logging, and the plug
point**; everything model-shaped lives in your fork.

*One step above a monolith:* write the monolith you'd have written anyway, minus the
boilerplate for tracking and testing.

```bash
make setup                                    # uv sync
uv run python -m dlt.train experiment=e0      # 20-step smoke run
./scripts/tb.sh                               # tensorboard over every run
uv run python scripts/runs.py --sort val/loss # query your runs
```

## The four assumptions

That's the honest statement of what this works with:

1. Training is a loop over batches from a dataloader.
2. Each step yields a scalar to minimise — or you take manual control.
3. Parameters are updated by `torch.optim` optimizers.
4. Progress is measured in steps.

`batch` is never inspected, model outputs are a free dict, and `training_step` returns
arbitrary scalars. Anything satisfying those four works, anticipated or not.

## Adding a model costs two files

```
src/dlt/project/jepa.py      # subclass TaskModule, write three methods
configs/model/jepa.yaml      # _target_ + hyperparameters
```

```bash
uv run python -m dlt.train model=jepa data=your_data
```

Untouched: the Trainer, TensorBoard wiring, text logging, checkpointing, and the test
suite — `tests/test_contracts.py` picks the new config up automatically and checks that
it instantiates, steps, produces a finite scalar loss, and populates gradients. **You
write no test code.**

See [ARCHITECTURE.md](ARCHITECTURE.md) for the contracts.

## Layout

```
src/dlt/
├── core/          # infra you rarely touch
│   ├── base.py         TaskModule + DataModule ABCs
│   ├── trainer.py      the loop + TrainState
│   ├── tracking.py     TensorBoard + JSONL + console + run_meta
│   ├── callbacks.py    checkpoint, early stop, LR, grad stats
│   ├── checkpoint.py   save/load/resume, pluggable weight format
│   └── utils.py        seed, device move, precision, schedules
├── project/       # yours — replace on fork
│   ├── mlp.py          two-layer net on random tensors (smoke/contract test)
│   ├── forecast.py     windowed timeseries forecasting
│   └── pinn.py         du/dx = -u, gradient loss + adaptive weighting
├── train.py
└── eval.py
```

## Observability

Three channels, all rank-zero-only, all toggleable in `configs/tracking/default.yaml`.
TensorBoard isn't always viewable — the other two are what keep a headless run legible.

```
outputs/<exp>/<timestamp>_<hash6>/
├── .hydra/          config.yaml, overrides.yaml   (Hydra writes these; we don't duplicate)
├── tb/              scalars, histograms, hparams, the run's own config as text
├── ckpt/            best/ and last/, each with weights + optimizer + RNG + extra/
├── artifacts/       manual inference output — see below
├── train.log        rich on a TTY, plain text over SSH/SLURM so it stays greppable
├── metrics.jsonl    one JSON object per log step; jq-able, pandas-loadable
└── run_meta.json    git sha, dirty flag, status, final metrics
```

Runs are grouped by experiment, time-sortable, and the 6-char hash marks identical
configs. There is **no counter** — counters race under parallel launch and differ per
machine. There is also no shared index file: `scripts/runs.py` *derives* the table by
scanning, so there's no append-and-rewrite race to get wrong.

```bash
uv run python scripts/runs.py --exp pinn --sort val/loss
uv run python scripts/runs.py --status failed        # crashed runs are visibly crashed
uv run python scripts/aggregate_seeds.py outputs/pinn/<ts>_sweep   # mean ± std
```

Manual inference output goes in the run's `artifacts/`, so it can never drift from the
weights that produced it:

```python
from dlt.core.tracking import artifacts_dir
d = artifacts_dir("outputs/pinn/2026-08-03_14-22-05_a3f9c2", "rollout_h1000")
torch.save(traj, d / "traj.pt")
```

## dtype and amp are separate axes

```yaml
dtype: float32   # float32 | float64   → what the model lives in
amp:   none      # none | fp16 | bf16  → autocast on top of fp32
```

They're mutually exclusive, and the pair is validated at startup — so `float64 + bf16`
is a clear config error, not a failure deep in a backward pass. `float64` is not
exotic: models with second derivatives routinely diverge in fp32.

## Sweeps

```bash
uv run python -m dlt.train -m experiment=pinn seed=1,2,3,4,5
uv run python -m dlt.train -m model.hidden_dim=32,64,128
```

No HPO library is shipped. `train.py` returns the monitored metric, which is the entire
coupling surface a sweeper needs — adding Optuna later touches nothing in `src/`.

## Requirements

Python **≥3.12, <3.14** — the ceiling is real: hydra-core 1.3.4 breaks on 3.14, whose
argparse rejects Hydra's lazy `--shell-completion` help object. Fixed in hydra-core 1.4
(dev-only as of Aug 2026); lift the pin then. `.python-version` pins 3.13.

## Forking

`core/` is what you never rewrite; `project/` is what you replace immediately. Rename
the `dlt` package to your project's name — it's a find-and-replace, and it matters
because a top-level package named `src` collides with every other project doing the
same.

Fixes don't propagate between forks. That's the deliberate trade: five forks, one
tracking bug, five fixes — in exchange for never fighting an abstraction that was built
for someone else's model.
