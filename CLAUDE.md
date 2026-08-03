# CLAUDE.md

Guidance for Claude Code working in this repository.

## What this is

A minimal, **fork-first** PyTorch training template. It provides the loop, the logging
and the plug point; everything model-shaped lives in the fork. ~1,100 lines total.

`src/dlt/core/` is infrastructure that is rarely touched. `src/dlt/project/` is example
code the user replaces. **Default to solving a problem in `project/` or in config, not
by adding machinery to `core/`.** If a change to `core/` seems necessary, say why the
existing seams don't cover it before writing it.

## Commands

```bash
uv sync --extra dev                          # install (Python >=3.12,<3.14)
uv run pytest tests/ -q                      # 61 tests
uv run ruff check src tests scripts          # lint (must be clean)
uv run ruff format src tests scripts
uv run python -m dlt.train experiment=e0     # ~20-step smoke run
uv run python -m dlt.train experiment=e0 debug=overfit   # first thing to try when broken
```

There is no compatible GPU on the development machine (sm_61 vs a torch build starting
at sm_75), so **verify on CPU** with `trainer.device=cpu` and do not claim GPU, DDP or
`torch.compile` paths are tested.

## Architecture in one paragraph

Two ABCs in `core/base.py`: `TaskModule` (`training_step`, `validation_step`,
`configure_optimizers`) and `DataModule` (`setup`, `train_dataloader`,
`val_dataloader`). The Trainer never inspects a batch — it only moves it to the device
via duck-typed `.to()`. Steps return a dict containing `"loss"`; every other scalar in
that dict is logged automatically. Adding a model is two files: one in `project/`, one
config in `configs/model/`.

Read `ARCHITECTURE.md` before changing `core/`. Read `USAGE.md` before changing a
user-facing workflow. Read `UNRESOLVED.md` at the start of a session — it lists what is
untested versus what is deliberately omitted, so you don't "fix" a decision.

## Non-obvious constraints

Each of these was a bug at some point. Breaking one is silent, not loud.

- **Anything that reads `.grad` must use the `on_before_optimizer_step` callback hook.**
  `on_train_batch_end` fires after `clip_and_step()` calls `zero_grad(set_to_none=True)`,
  so gradients are already `None` and stats read as `0.0`.
- **Resume must go through `Trainer.fit(..., resume=...)`.** `fit()` builds the
  optimizers, so loading a checkpoint before it populates optimizers that are then
  discarded — the run continues with a cold optimizer and no error.
- **`global_step` counts optimizer steps, not micro-batches.** With `grad_accum=4`,
  counting micro-batches would shift every schedule and logged x-axis by 4x.
- **`paths.output_root` must stay a plain relative string.** `hydra.sweep.dir` resolves
  before `HydraConfig` exists, so any `${hydra:...}` in that chain fails on `--multirun`
  while working fine for single runs.
- **`hydra.job_logging`'s file handler must name `${hydra.runtime.output_dir}`.** With
  `chdir: false`, the default writes relative to CWD and dumps every log in the repo root.
- **The `run_hash` resolver needs `use_cache=True`.** Without it each interpolation
  re-evaluates and one run scatters across several directories.
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

## Measured library behaviour

Verified on torch 2.13.0 / safetensors 0.8.0 — do not re-derive from memory:

- safetensors round-trips `complex64` but raises `KeyError` on `complex128`. Hence the
  pluggable `WeightFormat` and `checkpoint.format=torch` escape hatch.
- `add_histogram` does **not** reject complex tensors — it silently casts to real and
  discards the imaginary part with a `ComplexWarning`. `TensorBoardLogger` splits complex
  into abs/real/imag deliberately.
- hydra-core 1.3.4 breaks on Python 3.14 (argparse rejects its lazy `--shell-completion`
  help object). Hence `requires-python = ">=3.12,<3.14"`. Lift when hydra-core 1.4 is
  stable.

## Conventions

- `project/` is organised **by example, not by layer** — each file holds its model, data
  and loss together so the user can delete one without touching three.
- **Callbacks observe; they never own the update.** Anything that changes results
  (augmentation, EMA of a target encoder, loss weighting) belongs in `training_step` or
  `collate_fn`. This is a deliberate trade, not an oversight.
- Config groups are swapped with `group=name`; experiment files use `# @package _global_`
  to override across groups.
- `tests/test_contracts.py` is auto-parametrized over every config in `configs/model/`.
  A new model config needs **one line** in `MODEL_DATA` to pair it with a datamodule and
  gets its contract tests for free — do not hand-write per-model tests.
- Prefer a test that asserts something falsifiable over one that asserts "it ran". The
  PINN example exists because `e^(-x)` gives a metric that cannot be gamed.

## Code graph

`graphify-out/graph.json` is a committed structural index (465 nodes, 1092 edges) for
locating code without reading files. It is **stale after any code change** — refresh it
with the structural, LLM-free command:

```bash
graphify update . --no-cluster        # ~2s, deterministic, nothing leaves the machine
graphify explain src/dlt/core/trainer.py
graphify god-nodes
```

Never run `graphify extract`, `label` or `cluster-only` without an explicit `--backend`:
bare, graphify selects one from the environment (AWS Bedrock if `AWS_PROFILE` is set),
which bills a real account and ships the code off-box.

## Working style for this repo

- Run the code before claiming it works. Several bugs here were invisible to inspection
  and obvious on execution.
- When a test passes, check it passed for the right reason. One resume test passed only
  because the run exited before touching the optimizer.
- State untested paths explicitly rather than implying full coverage.
