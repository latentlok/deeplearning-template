# Architecture

Read this before extending. It is short on purpose.

## The two contracts

`abc.ABC` + `@abstractmethod`, so forgetting a method is a loud `TypeError` at
construction rather than a confusing failure mid-training. **The ABCs fix only the
boundary; every internal is yours.**

### `TaskModule` — `engine/base.py`

```python
class TaskModule(nn.Module, ABC):
    manual_optimization: bool = False   # you own backward/step entirely
    eval_requires_grad:  bool = False   # eval NOT wrapped in no_grad (physics losses)

    def training_step(self, batch, state) -> dict[str, Tensor]   # must contain "loss"
    def validation_step(self, batch, state) -> dict[str, Tensor]
    def configure_optimizers(self) -> OptimSpec                  # 1..N optimizers

    # optional, no-op by default
    def on_data_ready(self, datamodule) -> None                  # data statistics -> buffers
    def rollout_step(self, batch, state) -> dict | None          # free-running eval
    def save_extra(self, dir) / load_extra(self, dir)            # tokenizers, vocabs
```

### `DataModule` — `engine/base.py`

```python
class DataModule(ABC):
    def setup(self, stage: str) -> None
    def train_dataloader(self) -> DataLoader | dict[str, DataLoader]
    def val_dataloader(self)   -> DataLoader | dict[str, DataLoader] | None

    # optional: define these and resume restores dataloader position exactly
    def state_dict(self) / load_state_dict(self, sd)
```

Everything else — losses, metrics, schedulers, weighting — is a plain object built by
`hydra.utils.instantiate` with `_target_`. No ABC required.

## Why the signatures look like this

Each one is load-bearing for a real case, not speculative generality.

**`batch` is opaque.** The Trainer never indexes it; it only calls a duck-typed
`move_to_device`. Dicts, tuples, and graph batch objects all work — and two-level
sampling (pick a geometry or series, subsample its points) stays a `Dataset` concern
rather than a pipeline concern.

**`training_step` returns arbitrary scalars.** Neural-ODE NFE counts, per-term physics
losses, adaptive loss weights and teacher-forcing ratios all ride along in the same
dict and get logged automatically. Return `"batch_size"` to control metric weighting —
the default infers dim 0, which for two-level sampling means *items*, not points.

**Both loaders may be dicts.** Keys become metric namespaces: `val/h24/mae`,
`val/256/l2`. This is what multi-horizon forecasting and multi-resolution evaluation
need; a single-loader contract would force a Trainer edit for the most standard
experiment in either field.

**`eval_requires_grad`** exists because a blanket `torch.no_grad()` in evaluation makes
PDE residuals uncomputable. The Trainer also **never** uses `torch.inference_mode()`,
which is stricter and taints tensors against re-entering autograd.

**`manual_optimization` is a ramp, not a cliff.** Flipping it keeps the Trainer's
services callable — `self.trainer.backward(loss)` and
`self.trainer.clip_and_step(opt)` — so taking control of one thing doesn't cost you AMP
scaling, gradient accumulation, clipping and DDP sync all at once.

## The one place data reaches the model

`on_data_ready(datamodule)` is called once inside `fit()`, after `setup("fit")` and
**before** the weights move to the device or a checkpoint loads. It exists for
statistics the model must *own* — normalisation bounds, a channel mean/std, a vocab
size — which belong in **buffers**, so they are written into every checkpoint and
restored with the weights.

```
train split → utils/stats.py → stats.json → DataModule.stats → model buffers → checkpoint
```

Two ordering decisions make that chain safe. Resume loads *after* the hook, so
checkpointed statistics beat whatever today's data reports. And `eval.py` never calls
the hook at all: an evaluation takes its bounds from the checkpoint, because
recomputing them from whatever is mounted is precisely the desync being prevented —
one that leaves predictions plausible rather than raising.

Statistics are **read from a file, never computed in `setup()`**. Scanning costs
minutes per run, and worse, the val split would normalise against different numbers
than the train split.

## The loop

`engine/trainer.py`. One concrete class with small overridable methods — subclass and
override `train_step` alone. No ABC hierarchy, which would force every trainer to
reimplement the skeleton.

It is **step-first**. Epochs are derived, and `len(dataloader)` is never called, so
streaming / `IterableDataset` training works without inventing a fake epoch.

Compile and distributed wrapping are isolated in one overridable method, `wrap()`, so a
fork can swap DDP for FSDP2 without reimplementing the loop. Two constraints hold it
together, both of which fail *silently* when broken: whatever `wrap()` returns must own
the forward — DDP and `torch.compile` act only on graphs built inside their own
`forward`, so calling `training_step` directly means no all-reduce and no tracing — and
`wrap()` must precede `configure_optimizers()`, because FSDP2 swaps every `Parameter`
for a sharded `DTensor` and an optimizer built first would hold tensors that never get
a gradient.

Four things gradient accumulation must get right, all handled:

- loss divided by `grad_accum` before backward
- `global_step` counts **optimizer** steps, not micro-batches — otherwise every
  schedule and every logged x-axis is off by `grad_accum`
- schedulers step on optimizer steps
- DDP uses `no_sync()` on micro-steps `1..N-1` to skip redundant all-reduce

Effective batch size (`batch_size × grad_accum × world_size`) is logged at startup,
because that's the number people get wrong and then can't reproduce.

## Callbacks observe; they never own the update

Checkpointing, early stopping, LR logging and gradient statistics are side-effects. The
optimisation logic stays in the Trainer and the TaskModule, so no callback can silently
change your results.

Stated as a trade rather than sold as a virtue: **mixup, cutmix and adversarial
perturbation cannot be callbacks.** They belong in `collate_fn` or `training_step`.
Likewise EMA of a target encoder — for JEPA-style methods that update *is* the
algorithm, and hiding it in a hook is exactly what this rule prevents.

## What a run leaves on disk

```
outputs/<exp>/<timestamp>_<hash6>/
    .hydra/            the exact composed config, written by Hydra
    train.log  metrics.jsonl  tb/  run_meta.json
    ckpt/
        last/               rolling resume point, overwritten every `last_every`
        step_00001000/      permanent snapshot every `every_steps`, pruned to `keep_last`
        best/               full state at the best metric — resume from here
        best_weights/       the same weights, no optimizer state — ship this
    eval/<timestamp>/  every `eval.py` run against this checkpoint
    artifacts/         yours, via engine.tracking.artifacts_dir
```

`best/` and `best_weights/` are both written because they answer different questions:
one continues training, the other is loaded for inference. Keeping only the full one
drags optimizer moments into every deployment; keeping only the weights means a
crashed run cannot pick up from its best point.

Step snapshots are permanent and step-tagged, so a divergence noticed at step 40k can
be traced back through steps `last/` has long since overwritten. `keep_last` bounds
what that costs.

Evaluation writes **into the run it evaluated**, not a global `outputs/eval/` tree —
`configs/eval.yaml` derives the directory from `ckpt` through the `ckpt_run_dir`
resolver. A parallel eval tree is orphaned the moment an experiment has two runs: the
numbers are real, and nothing on disk says which weights produced them.

## Extension points, all open

| Want to change | Do this |
|---|---|
| the update rule | subclass `Trainer`, override `train_step` |
| the distributed strategy | subclass `Trainer`, override `wrap` (FSDP2, custom DDP) |
| a tracking backend | subclass `Logger` in `engine/tracking.py` |
| checkpoint encoding | subclass `WeightFormat`, point `checkpoint.format` at it |
| observation | add a `Callback` |
| anything model-shaped | write a file in `models/` |
| anything data-shaped | write a file in `dataset/` |

The `WeightFormat` seam exists because `safetensors` supports a fixed dtype set —
measured, not assumed: `complex64` round-trips, **`complex128` raises `KeyError`**. A
model at `dtype: float64` with complex spectral weights therefore needs either
`format: torch` or a custom format. `engine/` stays general; the fork adds what it needs.

Related measured gotcha: `add_histogram` does **not** reject complex tensors — it
silently casts to real and discards the imaginary part with only a `ComplexWarning`. A
silent half-truth in diagnostics is worse than a crash, so `TensorBoardLogger` splits
complex into `abs`/`real`/`imag`.

## Config

Hydra. Run directories are `outputs/<exp>/<timestamp>_<hash6>` via a `run_hash`
resolver registered **before** `@hydra.main` composes, with `use_cache=True` — without
the cache each interpolation re-evaluates and one run scatters across several
directories.

Two hashes do two jobs. The directory name uses a cheap hash of the override string,
which is all that's available at `hydra.run.dir` interpolation time; edit a YAML in
place and it won't change. `run_meta.json` therefore records the authoritative hash of
the fully-resolved config.

`hydra.job.chdir` is `false`, so relative data paths don't break inside the run dir.
Two consequences: `job_logging`'s file handler must name `${hydra.runtime.output_dir}`
explicitly or every run's log lands in the repo root; and `paths.output_root` must stay
a plain relative string, because `hydra.sweep.dir` is resolved *before* `HydraConfig`
exists and any `${hydra:...}` there fails on `--multirun`.

Hydra's basic launcher runs multirun jobs sequentially **in one process**, so dtype and
seed are set *unconditionally* every job — anything conditional leaks between jobs.
