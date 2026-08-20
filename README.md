# dlt — a minimal, fork-first PyTorch training template

Train, track, log. Nothing else.

Every deep learning project rewrites the same scaffolding: the training loop, the
checkpoints, the metric logging, the config plumbing. None of it is the interesting
part, and all of it has to work before the interesting part can be tried. This repo is
that scaffolding, already written and already tested — fork it, drop in your model and
your data, and start running experiments on day one. It deliberately does *not* try to
abstract over what a model is: you write the model you'd have written anyway.

## Architecture

Your code lives at the top level. The machinery is hidden in `engine/`.

```
train.py  eval.py     the two entrypoints
models/               your models — one file each
dataset/              your dataloaders — loader.py reads $DL_DATA/{train,val}.zarr
configs/              one directory per config group; experiment/ holds saved recipes
utils/                dataset statistics, run tables, seed aggregation
engine/               the loop, checkpointing, logging. You should never need to open it.
outputs/<exp>/<run>/  everything a run produced: logs, metrics, tb/, ckpt/, eval/
```

You implement two classes and the Trainer handles the rest:

- **`TaskModule`** — `training_step`, `validation_step`, `configure_optimizers`.
  A step returns a dict containing `"loss"`; every other scalar in it is logged for free.
- **`DataModule`** — `setup`, `train_dataloader`, `val_dataloader`.

The Trainer never looks inside a batch, so any batch shape works. The dataset lives
**outside** the repo — set `DL_DATA` once and every config follows it.

## Use it

Needs [uv](https://docs.astral.sh/uv/) and Python ≥3.12, <3.14 (uv fetches it).

```bash
uv sync --extra dev                     # install
uv run pytest tests/ -q                 # 94 tests, all should pass
uv run python train.py experiment=e0    # ~20-step smoke run
```

Then:

```bash
uv run python train.py experiment=e1                  # train on your own zarr data
uv run python train.py experiment=e0 trainer.max_steps=5000 optim.lr=1e-4
uv run python eval.py ckpt=outputs/e0/<run>/ckpt/best_weights
```

Adding a model is two files: one in `models/`, one config in `configs/model/`. Copy
`models/mlp.py` — it is the smallest complete example. `forecast.py` and `pinn.py` show
harder cases (rollout, gradient losses); delete what you don't need.

## Next

- **[USAGE.md](USAGE.md)** — experiments, configs, checkpoints, resuming, sweeps,
  adding your own model.
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the two contracts and why each decision is
  the way it is.

## Licence

[MIT](LICENSE).
