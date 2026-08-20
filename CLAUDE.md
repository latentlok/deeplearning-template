# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A minimal, **fork-first** PyTorch training template. It provides the loop, the logging
and the plug points; everything model-shaped and data-shaped lives at the top level.
~3,300 lines including docstrings and comments; ~2,400 of them (`engine/`, `utils/`,
the entrypoints) you never rewrite.

```
train.py  eval.py   entrypoints. @hydra.main must stay HERE (see below).
models/             the user's models. One TaskModule per file.
dataset/            the user's dataloaders. loader.py reads a zarr group from $DL_DATA.
utils/              stats.py, runs.py, aggregate_seeds.py (offline) + normalize.py
configs/            one directory per config group
engine/             the orchestration. Rarely touched.
```

**Default to solving a problem in `models/`, `dataset/` or in config, not by adding
machinery to `engine/`.** If a change to `engine/` seems necessary, say why the
existing seams don't cover it before writing it.

The dataset lives **outside** the repo at `$DL_DATA` (`paths.data_root`). Never write
code that assumes data is in the working tree.

## Commands

```bash
uv sync --extra dev                      # install (Python >=3.12,<3.14)
uv run pytest tests/ -q                  # 94 tests
uv run ruff check engine models dataset utils tests train.py eval.py   # must be clean
uv run ruff format engine models dataset utils tests train.py eval.py
uv run python train.py experiment=e0     # ~20-step smoke run
uv run python train.py experiment=e0 debug=overfit   # first thing to try when broken
uv run python eval.py ckpt=outputs/e0/<run>/ckpt/best_weights
uv run python utils/stats.py --root $DL_DATA         # write stats.json
```

**This is a template, and it has no target machine.** Never write anything here that
assumes a particular card, driver or accelerator count, and never hardcode a device.
`trainer.device: auto` already resolves to `cuda:LOCAL_RANK` when a GPU is present and
falls back to `cpu` when one is not (`resolve_device` in `engine/utils.py`), so code
and docs alike should assume a current GPU and let the fallback handle the rest.

Testing distributed does not need a GPU either — the `gloo` backend runs it on CPU:

```bash
uv run torchrun --nproc_per_node=2 train.py experiment=e0 trainer.device=cpu
```

Exercised that way: two-rank DDP, `no_sync()` under `grad_accum`, rank-zero gating,
`torch.compile`, `amp=bf16`/`fp16`, and FSDP2 through `Trainer.wrap()`. The `nccl`
backend, multi-GPU device placement and `GradScaler` on fp16 hardware have not been
exercised — state that rather than implying coverage.

Unverified in `dataset/loader.py`: `ZarrData` with `num_workers > 0`.
`configs/data/zarr.yaml` defaults to `0`; zarr reopens its chunk store per worker, which
is the thing to watch there.

**Data is zarr only.** The `.npy` / `.pt` loader is gone — `loader.py` reads a zarr
*group* of named variables and nothing else. A zarr 3 `Array` has **no `__len__`**; use
`.shape[0]`, or you get a bare `TypeError` (this is exactly what silently broke the old
`_open` zarr branch). Indexing a `(T,)` array with an int gives a 0-d array, not a
scalar, hence `np.atleast_1d` in `row()`.

## Architecture in one paragraph

Two ABCs in `engine/base.py`: `TaskModule` (`training_step`, `validation_step`,
`configure_optimizers`) and `DataModule` (`setup`, `train_dataloader`,
`val_dataloader`). The Trainer never inspects a batch — it only moves it to the device
via duck-typed `.to()`. Steps return a dict containing `"loss"`; every other scalar in
that dict is logged automatically. Adding a model is two files: one in `models/`, one
config in `configs/model/`.

Read `ARCHITECTURE.md` before changing `engine/`. Read `USAGE.md` before changing a
user-facing workflow. Read *Deliberately omitted* below before "fixing" anything that
looks missing — most of it is a decision.

## Non-obvious constraints

Each of these was a bug at some point. Breaking one is silent, not loud.

- **`@hydra.main` must stay in the root `train.py` / `eval.py`.** Hydra derives its
  config search path from the module owning the decorated function; only when that
  module is `__main__` does it use the file's directory. Move the decorator into
  `engine/` and it looks for a `configs` *package* and dies with "Primary config module
  'configs' not found". `engine/train.py::run` holds the body.
- **Anything that reads `.grad` must use the `on_before_optimizer_step` callback hook.**
  `on_train_batch_end` fires after `clip_and_step()` calls `zero_grad(set_to_none=True)`,
  so gradients are already `None` and stats read as `0.0`.
- **Resume must go through `Trainer.fit(..., resume=...)`.** `fit()` builds the
  optimizers, so loading a checkpoint before it populates optimizers that are then
  discarded — the run continues with a cold optimizer and no error.
- **Data statistics reach the model only through `on_data_ready`, and only in `fit()`.**
  They belong in buffers so they ride inside the checkpoint. `eval.py` must never call
  it: an evaluation takes its bounds from the checkpoint, not from whatever data is
  mounted. Resume loads *after* the hook, so checkpointed values win.
- **Normalisation lives in the model (`utils/normalize.py`), never in the dataloader.**
  `Normalizer` holds mean/std as buffers, so `forward()` takes raw units and returns raw
  units and inference needs only the checkpoint. Normalising in `__getitem__` instead
  would leave the checkpoint unable to denormalise its own predictions.
- **`stats.json` is keyed by variable name, not column position.** A positional list
  silently pairs a variable with another variable's mean the moment `data.inputs` is
  reordered or subset. `_assemble` concatenates in config order, matching `row()`.
- **`Normalizer.fit` clamps `std` to `1e-8`.** A constant variable has `std == 0` and
  would normalise every sample to `inf`.
- **Never compute dataset statistics in `DataModule.setup()`.** They are read from
  `stats.json`, written once by `utils/stats.py`. Computing them per run costs minutes
  and makes the val split normalise against different numbers than train.
- **`global_step` counts optimizer steps, not micro-batches.** With `grad_accum=4`,
  counting micro-batches would shift every schedule and logged x-axis by 4x.
- **Nothing in `configs/paths/` may chain to `${hydra:...}`.** `hydra.sweep.dir`
  resolves before `HydraConfig` exists (so `--multirun` dies while single runs work),
  and `hydra.compose()` in the tests has no `HydraConfig` at all.
- **`hydra.job_logging`'s file handler must name `${hydra.runtime.output_dir}`.** With
  `chdir: false`, the default writes relative to CWD and dumps every log in the repo root.
- **The `run_hash` resolver needs `use_cache=True`.** Without it each interpolation
  re-evaluates and one run scatters across several directories. `ckpt_run_dir` is a pure
  function of its argument and needs no cache.
- **Checkpoint snapshot names are zero-padded** (`step_00001000`). Pruning sorts by
  name, and unpadded names put `step_9` after `step_10`.
- **Set dtype and seed unconditionally every run.** Hydra's basic launcher runs multirun
  jobs sequentially in one process, so conditional global state leaks between jobs.
- **`torch.set_default_dtype` must precede model construction**, or the model is fp32
  while the data is fp64 and you get a confusing matmul error rather than a config error.
- **Never use `torch.inference_mode()`** in evaluation. It is stricter than `no_grad` and
  taints tensors against re-entering autograd; models with physics losses set
  `eval_requires_grad = True` and need real gradients during eval.
- **Freezing a submodule needs a `train()` override**, not just `.eval()` in `__init__` —
  the Trainer calls `.train()` every step, so BatchNorm statistics otherwise drift while
  the weights stay frozen.
- **Whatever `Trainer.wrap()` returns must own the forward.** DDP and `torch.compile`
  only act on graphs built inside their own `forward`. Calling `raw.training_step()`
  directly walks past both: DDP's reducer never runs `prepare_for_backward`, so nothing
  is all-reduced and the ranks silently train into different models, and compile traces
  zero graphs. Hence `_StepWrapper`, and hence `train_step` calling `self.module(...)`.
- **`wrap()` must run before `configure_optimizers()`.** FSDP2 replaces every
  `Parameter` with a sharded `DTensor`; an optimizer built first holds the pre-shard
  objects, which never receive a gradient. The run trains, logs a loss and updates
  nothing.
- **Move modules with `cast_module`, never `module.to(dtype=...)`.** `Module._apply`
  casts anything `is_floating_point() OR is_complex()`, so `.to(float32)` flattens a
  `complex64` weight to real and discards the imaginary part with only a warning.
- **Rank-gate logger *construction*, not just its calls.** `MultiLogger` drops writes
  off rank zero, but `SummaryWriter.__init__` creates its event file immediately, so
  every rank left a phantom run in `tb/`.

## Measured library behaviour

Verified on torch 2.13.0 / safetensors 0.8.0 — do not re-derive from memory:

- safetensors round-trips `complex64` but raises `KeyError` on `complex128`. Hence the
  pluggable `WeightFormat` and `checkpoint.format=torch` escape hatch.
- `add_histogram` does **not** reject complex tensors — it silently casts to real and
  discards the imaginary part with a `ComplexWarning`. `TensorBoardLogger` splits complex
  into abs/real/imag deliberately.
- `nn.Module.to(dtype)` casts complex parameters too — `_apply` converts anything
  `is_floating_point()` **or** `is_complex()` — so a real dtype silently destroys a
  complex weight. `cast_module` maps complex to the matching complex dtype instead.
- `fully_shard`'s default device mesh picks the *accelerator*, so on a one-GPU box it
  fails with `invalid device ordinal`. Pass an explicit `init_device_mesh("cpu", ...)`
  to exercise FSDP2 locally.
- A `DTensor` has no accessible storage pointer, so safetensors cannot save sharded
  weights. The existing `checkpoint.format=torch` / `WeightFormat` seam covers it.
- hydra-core 1.3.4 breaks on Python 3.14 (argparse rejects its lazy `--shell-completion`
  help object). Hence `requires-python = ">=3.12,<3.14"`. Lift when hydra-core 1.4 is
  stable — still `1.4.0.dev6` as of 2026-08-03, so the ceiling stays.

## Conventions

- **`models/` holds nothing data-shaped; `dataset/` holds nothing model-shaped.** The
  project is one dataset with many models tried against it.
- **`dataset/` vs `utils/` splits on when code runs**: per-batch hot path versus
  run-once offline analysis that writes a file. `utils/normalize.py` is the one
  exception — it runs per batch but is a shared building block rather than a model, so
  it sits with the other shared pieces.
- **Optimizer and scheduler come from `configs/optim/` and `configs/sched/`**, not from
  hardcoded calls in `configure_optimizers`. Both are `_partial_: true`, so hydra hands
  the model a factory it calls with `self.parameters()` (or the built optimizer). Model
  configs interpolate `optim: ${optim}` and `sched: ${oc.select:sched,null}` — plain
  `${sched}` raises when the group is null, which is the default. `lr` survives on the
  model constructor purely as the fallback for direct construction in tests.
- **Callbacks observe; they never own the update.** Anything that changes results
  (augmentation, EMA of a target encoder, loss weighting) belongs in `training_step` or
  `collate_fn`. This is a deliberate trade, not an oversight.
- The project is a **virtual uv project** (`[tool.uv] package = false`) — nothing is
  installed, code is imported from the repo root. `pythonpath = ["."]` in the pytest
  config is what makes tests see it.
- Config groups are swapped with `group=name`; experiment files use `# @package _global_`
  to override across groups.
- `tests/test_contracts.py` is auto-parametrized over every config in `configs/model/`.
  A new model config needs **one line** in `MODEL_DATA` to pair it with a datamodule and
  gets its contract tests for free — do not hand-write per-model tests.
- Prefer a test that asserts something falsifiable over one that asserts "it ran". The
  PINN example exists because `e^(-x)` gives a metric that cannot be gamed.

## Deliberately omitted — decisions, not gaps. Don't "fix" these.

- **No HPO / sweeper.** `train.py` returns the monitored metric, which is the whole
  coupling surface; adding Optuna later touches nothing in `engine/`.
- **No FSDP or model parallelism in `engine/`.** DDP is what ships. FSDP2 is *reachable*
  without engine edits — subclass `Trainer` and override `wrap()`; verified on gloo/CPU,
  weights update and ranks stay in sync. Sharded checkpoints need a `WeightFormat`.
- **No architecture-compatibility probing in `device: auto`.** On a mismatched box pass
  `trainer.device=cpu`.
- **No model-specific machinery in `engine/`** (FNO, GNO, Transolver…). Fork per model
  family; extend via `Trainer`, `Logger`, `WeightFormat`, `Callback`.
- **No `notebooks/explore_run.ipynb`.** A stub notebook is noise; `utils/runs.py` plus
  `metrics.jsonl` cover the real need.
- **No weights-only CLI flag.** `resume=<dir>` is an exact resume. Fine-tuning from
  someone else's weights is `load_checkpoint(..., weights_only=True)` behind an
  `init_from` argument on your model, so it stays configurable — recipe in `USAGE.md`.
- **No DataModule implements `state_dict()`/`load_state_dict()`.** The mechanism is
  tested; no shipped example uses it, so resume restores model and optimizer while the
  loader restarts from the top, and the Trainer says so. Only bites streaming runs.
- `EarlyStopping` and `GradStats` are implemented and tested but commented out in
  `configs/callbacks/default.yaml` — enable per experiment.

## Code graph

`.graphify/graph.json` is a committed structural index for
locating code without reading files. It is **stale after any code change** — refresh it
with the structural, LLM-free command:

```bash
make graph                              # graphify update . --no-cluster, then relocates
graphify explain engine/trainer.py --graph .graphify/graph.json
```

graphify always writes to `./graphify-out`; `make graph` moves the result into
`.graphify/` so the repo root stays clean.

Never run `graphify extract`, `label` or `cluster-only` without an explicit `--backend`:
bare, graphify selects one from the environment (AWS Bedrock if `AWS_PROFILE` is set),
which bills a real account and ships the code off-box.

## Working style for this repo

- Run the code before claiming it works. Several bugs here were invisible to inspection
  and obvious on execution.
- When a test passes, check it passed for the right reason. One resume test passed only
  because the run exited before touching the optimizer.
- State untested paths explicitly rather than implying full coverage.
