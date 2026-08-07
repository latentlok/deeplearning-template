# dlt — a minimal, fork-first PyTorch training template

Train, track, log. Nothing else.

You cannot design one abstraction that covers PINNs, JEPA, neural operators, neural
ODEs, forecasters and language models. Anything general enough to span them all is
worse than no abstraction — so this doesn't try.

It gives you **the loop, the logging, and the plug point**. Everything model-shaped
lives in your fork. *One step above a monolith:* you write the monolith you'd have
written anyway, minus the boilerplate for experiment tracking and testing.

It works with anything that satisfies four assumptions:

1. Training is a loop over batches from a dataloader.
2. Each step yields a scalar to minimise — or you take manual control.
3. Parameters are updated by `torch.optim` optimizers.
4. Progress is measured in steps.

`batch` is never inspected, model outputs are a free dict, and steps return arbitrary
scalars. Roughly 1,400 lines total, ~1,000 of which you never rewrite.

## Layout

Your code is at the top level. The orchestration is hidden in `engine/`.

```
train.py  eval.py     the two entrypoints
models/                your models — one file per model, each a TaskModule
dataset/               your dataloaders — loader.py reads $DL_DATA/{train,val}
utils/                 offline analysis: dataset statistics, run tables, seed aggregation
configs/               one group per directory; experiment/ holds saved recipes
engine/                the loop, checkpointing, logging. You should never need to open it.
outputs/<exp>/<run>/   everything a run produced: logs, metrics, tb/, ckpt/, eval/
```

The dataset itself lives **outside** the repo — set `DL_DATA` once and every config
follows it.

## Install

Requires **Python ≥3.12, <3.14** and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/latentlok/deeplearning-template.git
cd deeplearning-template
uv sync --extra dev
```

`.python-version` pins 3.13, and uv will fetch it if you don't have it. The upper bound
is real, not caution: hydra-core 1.3.4 breaks on Python 3.14, whose argparse rejects
Hydra's lazy `--shell-completion` help object. Lift it when hydra-core 1.4 ships stable.

Verify:

```bash
uv run pytest tests/ -q                # 74 tests
uv run python train.py experiment=e0   # ~20-step smoke run
```

The default `trainer.device: auto` uses CUDA when available. Note that
`torch.cuda.is_available()` returns True even for a GPU your torch build has no kernels
for — if you hit `CUBLAS_STATUS_ARCH_MISMATCH`, install a torch build matching your card
or pass `trainer.device=cpu`.

## Next

- **[USAGE.md](USAGE.md)** — running experiments, configs, checkpoints and resuming,
  evaluation, sweeps, adding your own model.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the two contracts, the loop, and why each
  design decision is the way it is.
- **[UNRESOLVED.md](UNRESOLVED.md)** — what is untested versus deliberately omitted.

## What ships

Three examples in `models/`, with their data in `dataset/examples.py` — delete the ones
you don't need:

| | |
|---|---|
| `mlp.py` | two-layer net; the fast smoke and contract test, and the file to copy |
| `forecast.py` | windowed forecasting: scaler-as-buffers, temporal split, teacher forcing vs free-running rollout, multi-horizon eval |
| `pinn.py` | `du/dx = -u` with a gradient loss and gradient-adaptive term weighting; converges against the analytic `e^(-x)` |

Plus `dataset/loader.py` — a real dataloader over `$DL_DATA/{train,val}` (`.npy`,
`.zarr`, `.pt`), which is the file you rewrite for your own data.

## Licence

[MIT](LICENSE).
