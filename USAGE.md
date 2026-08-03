# Usage

Everything here is verified against the shipped examples. For *why* the design looks
like this, see [ARCHITECTURE.md](ARCHITECTURE.md).

- [Running a training run](#running-a-training-run)
- [Configuration](#configuration)
- [Checkpoints and resuming](#checkpoints-and-resuming)
  - [Fine-tuning from a checkpoint](#fine-tuning-from-a-checkpoint-weights-only)
  - [Freezing layers](#freezing-layers)
- [Evaluation](#evaluation)
- [Sweeps and multirun](#sweeps-and-multirun)
- [Outputs and tracking](#outputs-and-tracking)
- [Finding and comparing runs](#finding-and-comparing-runs)
- [Adding your own model](#adding-your-own-model)
- [Precision: dtype and amp](#precision-dtype-and-amp)
- [Rollouts and manual inference](#rollouts-and-manual-inference)
- [Manual optimization](#manual-optimization)
- [Debugging](#debugging)
- [Distributed](#distributed)
- [Troubleshooting](#troubleshooting)

---

## Running a training run

```bash
uv run python -m dlt.train                      # defaults (mlp + synthetic data)
uv run python -m dlt.train experiment=e0        # a named experiment
uv run python -m dlt.train experiment=forecast
uv run python -m dlt.train experiment=pinn
```

Override anything from the command line using dotted paths:

```bash
uv run python -m dlt.train experiment=forecast \
    model.hidden_dim=128 \
    trainer.max_steps=5000 \
    data.batch_size=64 \
    seed=7
```

`make train EXP=pinn` and `make smoke` are shortcuts.

---

## Configuration

Config groups live in `configs/`. Each subdirectory is a group; the file you select
becomes that part of the config.

```
configs/
├── train.yaml          entrypoint: defaults list + globals (seed, dtype, amp, resume)
├── eval.yaml
├── model/              mlp.yaml  forecast.yaml  pinn.yaml
├── data/               synthetic.yaml  series.yaml  collocation.yaml
├── trainer/default.yaml
├── callbacks/default.yaml
├── tracking/default.yaml
├── paths/default.yaml
├── experiment/         e0.yaml  forecast.yaml  pinn.yaml
└── debug/              fast_dev.yaml  overfit.yaml
```

Swap a group with `group=name`:

```bash
uv run python -m dlt.train model=pinn data=collocation
```

### Experiment configs

An experiment file is the intended way to pin a whole setup. It uses
`# @package _global_` so it can reach across every group at once:

```yaml
# configs/experiment/my_run.yaml
# @package _global_
defaults:
  - override /model: forecast
  - override /data: series

exp_name: my_run
seed: 42
dtype: float32

model:
  hidden_dim: 128
trainer:
  max_steps: 20000
  val_every: 500
  monitor: val/h8/loss
```

```bash
uv run python -m dlt.train experiment=my_run
```

### Trainer options

```yaml
max_steps: 2000        # primary — the loop is step-first
max_epochs: null       # optional cap, checked at epoch boundaries
grad_accum: 1          # effective batch = batch_size x grad_accum x world_size
clip_grad: null
device: auto           # auto | cpu | cuda | cuda:0
compile: false         # torch.compile; off by default, it obscures tracebacks
log_every: 50          # steps
val_every: 200         # steps
rollout_every: 0       # steps; 0 disables
monitor: val/loss
monitor_mode: min      # min | max
```

Everything is in **steps**, not epochs. If you think in epochs, set
`val_every` to your steps-per-epoch.

---

## Checkpoints and resuming

### What gets written

Checkpointing is a callback (`configs/callbacks/default.yaml`). By default it writes
`last/` every `save_every` steps and at the end of the run, and `best/` whenever the
monitored metric improves:

```
outputs/e0/2026-08-03_14-43-01_b11a8d/ckpt/
├── best/
│   ├── model.safetensors    weights (fast, zero-copy, no pickle execution risk)
│   ├── state.pt             optimizers, schedulers, TrainState, RNG, dataloader state
│   └── extra/               non-tensor state via your save_extra hook
└── last/
    ├── model.safetensors
    ├── state.pt
    └── extra/
```

```yaml
callbacks:
  checkpoint:
    dirpath: ${hydra:runtime.output_dir}/ckpt
    save_every: 500        # steps; 0 disables periodic saves
    save_last: true
    save_best: true
    monitor: null          # null -> uses trainer.monitor
    fmt: safetensors       # safetensors | torch | a _target_ WeightFormat
```

### Exact resume

Point `resume` at a **checkpoint directory** (not a file). No `+` prefix is needed —
`resume` is declared in `train.yaml`:

```bash
uv run python -m dlt.train experiment=e0 \
    resume=outputs/e0/2026-08-03_14-43-01_b11a8d/ckpt/last
```

You'll see confirmation in the log:

```
[dlt.core.trainer][INFO] - resuming from outputs/e0/.../ckpt/last
[dlt.core.trainer][INFO] - resumed at step 500 (epoch 499)
```

Training then continues **from that step** toward `max_steps`. To train further, raise
`max_steps` — resuming at step 500 with `max_steps=500` exits immediately, which is
correct but looks like nothing happened:

```bash
uv run python -m dlt.train experiment=e0 \
    resume=outputs/e0/<run>/ckpt/last trainer.max_steps=5000
```

**What is restored:** model weights and buffers, every optimizer's state (Adam moments
and friends), scheduler state, `TrainState` (`global_step`, `epoch`, `samples_seen`,
`best_metric`, metrics), CPU and CUDA RNG state, and anything your `load_extra` hook
reads.

Resume is handled *inside* `Trainer.fit()`, after the optimizers it will actually use
have been constructed. Loading earlier would populate optimizers that `fit()` then
replaces — the run would continue with a cold optimizer, with no error to tell you.

**The one caveat — dataloader position.** Unless your `DataModule` implements
`state_dict()` / `load_state_dict()`, the loader restarts from the beginning of the
data. For epoch-style training on a finite dataset this rarely matters. For a streaming
run it does: you silently retrain the same prefix. The template says so rather than
pretending otherwise:

```
WARNING - resuming at step 40000 but the DataModule saved no state -- data position is
best-effort and the loader restarts from the beginning. Implement
DataModule.state_dict()/load_state_dict() to make it exact.
```

To make it exact, implement the pair on your DataModule; the Trainer round-trips
whatever you return through `state.pt`:

```python
class MyData(DataModule):
    def state_dict(self):
        return {"samples_consumed": self._consumed}

    def load_state_dict(self, sd):
        self._consumed = sd.get("samples_consumed", 0)
        # skip ahead in your stream accordingly
```

### Fine-tuning from a checkpoint (weights only)

An exact resume is not what you want for transfer or fine-tuning — there you want the
weights with a *fresh* optimizer and step counter. That is `weights_only=True`, which is
a Python-level call rather than a CLI flag.

The idiomatic place for it is your model's `__init__`, so it stays configurable and the
whole thing still runs through `dlt.train` unchanged:

```python
# src/dlt/project/mymodel.py
from pathlib import Path
from dlt.core.checkpoint import load_checkpoint


class MyModel(TaskModule):
    def __init__(self, hidden_dim: int = 64, lr: float = 1e-4,
                 init_from: str | None = None):
        super().__init__()
        self.net = ...
        self.lr = lr
        if init_from:                       # weights only: fresh optimizer, step 0
            load_checkpoint(Path(init_from), self, weights_only=True)
```

```yaml
# configs/model/mymodel.yaml
_target_: dlt.project.mymodel.MyModel
hidden_dim: 64
lr: 1e-4
init_from: null
```

```bash
uv run python -m dlt.train model=mymodel \
    model.init_from=outputs/pretrain/<run>/ckpt/best \
    model.lr=1e-5
```

`load_checkpoint(..., weights_only=True)` reads `model.safetensors` and runs your
`load_extra` hook, and skips `state.pt` entirely — no optimizer, no scheduler, no
`TrainState`. Contrast with `resume=`, which restores everything and continues the same
run. Use `resume=` to continue an interrupted run; use `init_from` to start a new one
from someone else's weights.

If the architectures differ, load the pieces yourself instead — `load_checkpoint`
expects the state dict to match:

```python
from safetensors.torch import load_file
sd = load_file(Path(init_from) / "model.safetensors")
self.encoder.load_state_dict({k[len("encoder."):]: v
                              for k, v in sd.items() if k.startswith("encoder.")})
```

### Freezing layers

Freezing is plain PyTorch — the template needs no support for it — but there are two
things to get right, and the second one bites silently.

**1. Freeze the parameters, and keep them out of the optimizer.**

```python
for p in self.encoder.parameters():
    p.requires_grad_(False)

def configure_optimizers(self):
    return OptimSpec.of(torch.optim.AdamW(
        filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr))
```

`requires_grad_(False)` alone is enough to stop the weights changing, but filtering the
optimizer avoids allocating momentum buffers for parameters that will never move.

**2. Override `train()` if the frozen part has BatchNorm or Dropout.**

The Trainer calls `.train()` on your module at the start of every step. So setting
`self.encoder.eval()` once in `__init__` does **not** stick — it is flipped back
immediately, and BatchNorm running statistics drift for the rest of training. The
weights stay frozen, so nothing errors; your frozen encoder just quietly stops producing
the same outputs it did at step 0.

The fix is the standard PyTorch idiom, and it works because the Trainer's `.train()`
call goes through your override:

```python
def train(self, mode: bool = True):
    super().train(mode)
    self.encoder.eval()        # frozen part stays in eval regardless
    return self
```

Verified behaviour with both pieces in place — frozen weights unchanged, BN statistics
unchanged, head trains normally, and the encoder is still in eval mode after 15 steps.
Without the `train()` override, BN statistics drift while the weights stay put.

Everything else already tolerates frozen parameters: gradient clipping skips parameters
with no gradient, DDP only reduces parameters with `requires_grad=True`, checkpoints save
and restore frozen weights like any other, and `tests/test_contracts.py` passes as long
as *something* remains trainable.

**Unfreezing partway through** (a common fine-tuning schedule) belongs in
`training_step`, where you have the step count:

```python
def training_step(self, batch, state):
    if state.global_step == self.unfreeze_at:
        for p in self.encoder.parameters():
            p.requires_grad_(True)
    ...
```

Note that parameters filtered out of the optimizer at construction stay out of it. To
unfreeze into a live optimizer you must also `add_param_group`, or build the optimizer
over all parameters up front and rely on `requires_grad` alone to gate the updates.

### Checkpoint format

`safetensors` is the default. It is fast and cannot execute arbitrary code on load, but
it supports a fixed dtype set — measured on safetensors 0.8.0 / torch 2.13.0:
`complex64` round-trips exactly, **`complex128` raises `KeyError`**. If you hit a dtype
it refuses, you get a message naming the fix rather than a silent conversion:

```bash
uv run python -m dlt.train experiment=mine callbacks.checkpoint.fmt=torch
```

`torch` format handles every dtype. If you need something specific — packing complex as
real, quantised weights, sharded writes — subclass `WeightFormat` and point the config
at it; nothing in `core/` changes:

```yaml
callbacks:
  checkpoint:
    fmt:
      _target_: myproject.ckpt.MyFormat
```

### Non-tensor state (tokenizers, vocabs, scalers)

`model.safetensors` holds tensors only. Anything else goes in `extra/` via two optional
hooks on your `TaskModule`, so it can never desync from the weights:

```python
def save_extra(self, directory: Path) -> None:
    self.tokenizer.save(directory / "tokenizer.json")

def load_extra(self, directory: Path) -> None:
    self.tokenizer = Tokenizer.load(directory / "tokenizer.json")
```

`load_extra` runs *before* the weights are loaded, so a tokenizer or scaler that the
model needs at construction is available in time.

Scaler statistics are better handled as `nn.Module` buffers — see `forecast.py`. Buffers
ride along inside `model.safetensors` automatically and need no hooks at all.

---

## Evaluation

`eval.py` loads a checkpoint (weights only — an eval should not inherit a stale
optimizer or step counter) and runs one pass:

```bash
uv run python -m dlt.eval ckpt=outputs/e0/<run>/ckpt/best
uv run python -m dlt.eval ckpt=outputs/e0/<run>/ckpt/last model=mlp data=synthetic
```

Use `step=rollout_step` to evaluate free-running instead of teacher-forced:

```bash
uv run python -m dlt.eval ckpt=outputs/forecast/<run>/ckpt/best step=rollout_step
```

The model and data configs must match what produced the checkpoint. Eval writes its own
run directory under `outputs/eval/` with its own `tb/`, `metrics.jsonl` and log.

---

## Sweeps and multirun

Hydra's `-m` flag runs the cross-product of comma-separated overrides:

```bash
uv run python -m dlt.train -m experiment=pinn seed=1,2,3,4,5
uv run python -m dlt.train -m model.hidden_dim=32,64,128 model.lr=1e-3,1e-4
```

Output lands in `outputs/<exp>/<timestamp>_sweep/{0,1,2,...}/`.

Then aggregate across seeds — a single-seed number is not a result:

```bash
uv run python scripts/aggregate_seeds.py outputs/pinn/2026-08-03_14-22-05_sweep
```

```
metric                n          mean           std
----------------------------------------------------
val/loss              5        9.5416       0.42295
val/mae               5         2.501      0.051047
```

**No HPO library is shipped.** `train.py` returns the monitored metric, which is the
entire coupling surface a sweeper needs — adding Optuna later touches nothing in `src/`.

Note that Hydra's basic launcher runs sweep jobs **sequentially in one process**. dtype
and seed are therefore set unconditionally on every job; if you add global state of your
own, set it unconditionally too or it will leak between jobs.

---

## Outputs and tracking

```
outputs/<exp>/<YYYY-MM-DD_HH-MM-SS>_<hash6>/
├── .hydra/          config.yaml, overrides.yaml  (Hydra writes these — not duplicated)
├── tb/              TensorBoard events
├── ckpt/            best/ and last/
├── artifacts/       your manual inference output
├── train.log        rich on a TTY, plain text otherwise
├── metrics.jsonl    one JSON object per logging step
└── run_meta.json    git sha, dirty flag, status, final metrics
```

Runs are grouped by experiment, time-sortable, and the 6-char hash marks identical
override sets. There is no counter — counters race under parallel launch and differ per
machine.

Three tracking channels, independently toggleable, all rank-zero only:

```yaml
# configs/tracking/default.yaml
tensorboard: true
jsonl: true
console: true
console_every: 50
```

TensorBoard isn't always viewable; `train.log` and `metrics.jsonl` are what keep a
headless run legible:

```bash
./scripts/tb.sh                       # tensorboard --logdir outputs
jq -s '.[-1]' outputs/e0/<run>/metrics.jsonl
tail -f outputs/e0/<run>/train.log
```

The TensorBoard **HPARAMS** tab is populated at the end of each run, so runs are
sortable by hyperparameter there. Each run also embeds its own resolved config under
**TEXT**, so a run is self-describing without cross-referencing the repo.

---

## Finding and comparing runs

There is no shared index file to corrupt — the table is *derived* by scanning:

```bash
uv run python scripts/runs.py                            # newest first
uv run python scripts/runs.py --exp pinn --sort val/loss
uv run python scripts/runs.py --where model.lr=0.005
uv run python scripts/runs.py --status failed            # crashed runs stay visible
uv run python scripts/runs.py --json
```

```
run                              status    git      val/loss   val/mae
-------------------------------  --------  -------  ---------  -------
pinn/2026-08-03_13-41-45_e5056b  finished  9f3a1c2  0.00054    0.0021
e0/2026-08-03_13-35-32_8b92a6    failed    9f3a1c2  -          -
```

`*` after the status marks a dirty working tree at launch. A run that crashes is
recorded as `failed` via the exception hook rather than silently vanishing, and its
traceback is in `train.log`.

`--where` matches the override string exactly as it was typed, so
`--where model.lr=0.005` finds a run launched with `model.lr=0.005` but not one launched
with `model.lr=5e-3`.

---

## Adding your own model

Two files. Nothing in `core/` changes.

**1.** `src/dlt/project/mymodel.py` — implement three methods:

```python
import torch
from torch import Tensor, nn
from dlt.core.base import DataModule, OptimSpec, TaskModule, TrainState


class MyModel(TaskModule):
    def __init__(self, hidden_dim: int = 64, lr: float = 1e-3):
        super().__init__()
        self.net = nn.Linear(hidden_dim, 1)
        self.lr = lr

    def training_step(self, batch, state: TrainState) -> dict[str, Tensor]:
        loss = nn.functional.mse_loss(self.net(batch["x"]), batch["y"])
        return {"loss": loss}                     # extra scalars here get logged too

    def validation_step(self, batch, state: TrainState) -> dict[str, Tensor]:
        pred = self.net(batch["x"])
        return {"loss": nn.functional.mse_loss(pred, batch["y"]),
                "mae": (pred - batch["y"]).abs().mean()}

    def configure_optimizers(self) -> OptimSpec:
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=1000)
        return OptimSpec.of(opt, sched, interval="step")   # "step" | "epoch"
```

**2.** `configs/model/mymodel.yaml`:

```yaml
_target_: dlt.project.mymodel.MyModel
hidden_dim: 64
lr: 1e-3
```

```bash
uv run python -m dlt.train model=mymodel data=synthetic
```

`tests/test_contracts.py` picks the new config up automatically and checks that it
instantiates, steps, produces a finite scalar loss, populates gradients on backward, and
survives a checkpoint round-trip. Add one line to `MODEL_DATA` in that file to pair it
with a datamodule; you write no other test code.

### Multiple optimizers, and schedulers that need metadata

```python
def configure_optimizers(self):
    return OptimSpec(
        optimizers=[torch.optim.Adam(self.gen.parameters(), lr=2e-4),
                    torch.optim.Adam(self.disc.parameters(), lr=1e-4)],
        schedulers=[SchedulerSpec(sched, interval="epoch", monitor="val/loss")],
    )
```

`interval` matters: cosine-with-warmup steps per optimizer step, classic decay per
epoch, and `ReduceLROnPlateau` needs `monitor` set to a metric key.

### Custom data

`batch` is never inspected by the Trainer — it is only moved to the device via a
duck-typed `.to()`. Dicts, tuples and graph batch objects all work, so two-level
sampling (pick an item, subsample its points) stays inside `Dataset.__getitem__`:

```python
def __getitem__(self, i):
    geo = self.geometries[i]
    idx = torch.randperm(geo.n_points)[:self.n_sample]
    return {"coords": geo.coords[idx], "target": geo.target[idx]}
```

If you want metrics weighted by *points* rather than *items*, return
`{"batch_size": n_points, ...}` from your step to override the default inference.

### Multiple validation loaders

Return a dict and the keys become metric namespaces — this is how multi-horizon
forecasting and multi-resolution evaluation work:

```python
def val_dataloader(self):
    return {"h8": loader_8, "h24": loader_24}   # -> val/h8/mae, val/h24/mae
```

`train_dataloader` accepts a dict too, for multi-task or mixture-of-corpora training.

---

## Precision: dtype and amp

These are **orthogonal** axes, not one setting. `dtype` is what the model lives in;
`amp` is an autocast wrapper on top of fp32.

```bash
uv run python -m dlt.train experiment=mine dtype=float32 amp=bf16
uv run python -m dlt.train experiment=pinn  dtype=float64 amp=none
```

| | |
|---|---|
| `dtype` | `float32` \| `float64` |
| `amp` | `none` \| `fp16` \| `bf16` — requires `dtype=float32` |

The pair is validated at startup, so a conflict is an immediate config error:

```
ValueError: amp='bf16' requires dtype='float32' (autocast wraps fp32); got
dtype='float64'. For fp64 runs use amp='none'.
```

`float64` is not exotic — models with second derivatives routinely diverge in fp32, and
`create_graph=True` compounds it. Consumer GPUs run fp64 at 1/32–1/64 throughput, so it
is fine for small physics models and painful at scale.

Per-module overrides need no template support: call `.double()` or `.to(dtype)` in your
module's `__init__`. Global default, local freedom.

---

## Rollouts and manual inference

Free-running evaluation is expensive and sequential, so it has its own cadence separate
from cheap teacher-forced validation:

```python
def rollout_step(self, batch, state) -> dict[str, Tensor] | None:
    ...   # autoregressive; returns plain scalars like any other step
```

```bash
uv run python -m dlt.train experiment=forecast trainer.rollout_every=1000
```

Metrics arrive namespaced `rollout/...`, or `rollout/h24/...` with multiple loaders.

For long-horizon testing, write your own script — it is too model-specific to
generalise — and put the output in the run's `artifacts/` so it can never drift from the
weights that produced it. The helper works outside Hydra, i.e. from a notebook:

```python
from dlt.core.tracking import artifacts_dir

d = artifacts_dir("outputs/pinn/2026-08-03_14-22-05_a3f9c2", "rollout_h1000")
torch.save(trajectory, d / "traj.pt")
```

---

## Manual optimization

Set `manual_optimization = True` and the Trainer stays out of the way — but its services
remain callable, so taking control of one thing doesn't cost you AMP scaling, gradient
clipping and DDP sync:

```python
class MyModel(TaskModule):
    manual_optimization = True

    def training_step(self, batch, state):
        loss_a, loss_b = self.compute(batch)
        self.trainer.backward(loss_a)
        self.trainer.clip_and_step(self.trainer.optimizers[0])
        self.trainer.backward(loss_b)
        self.trainer.clip_and_step(self.trainer.optimizers[1])
        return {"loss": (loss_a + loss_b).detach()}
```

You still get all the tracking, checkpointing, provenance and contract tests. Use this
for alternating updates, gradient surgery, or anything else that doesn't fit one
backward per step. In manual mode, dividing by `grad_accum` is your responsibility.

To change the loop itself rather than one step, subclass the Trainer and override one
method:

```python
class MyTrainer(Trainer):
    def train_step(self, stream, acc): ...
```

```yaml
trainer:
  _target_: myproject.MyTrainer
```

---

## Debugging

```bash
uv run python -m dlt.train experiment=mine debug=fast_dev   # 2 steps, no checkpoints
uv run python -m dlt.train experiment=mine debug=overfit    # overfit a few samples
```

**Run `debug=overfit` first when something is wrong.** It trains on a handful of
noiseless samples — if the loss does not approach zero, the bug is in your model or
loss, not your hyperparameters.

Gradient and weight histograms are available but off by default, since they are
expensive. Uncomment in `configs/callbacks/default.yaml`:

```yaml
grad_stats:
  _target_: dlt.core.callbacks.GradStats
  every: 500
```

Other callbacks: `EarlyStopping` (patience on the monitored metric), `NaNGuard` (stops
on a non-finite loss), `Timer`, `LRMonitor`.

For a full Hydra traceback rather than its abbreviated one:

```bash
HYDRA_FULL_ERROR=1 uv run python -m dlt.train experiment=mine
```

---

## Distributed

The Trainer is rank-aware throughout — logging, checkpointing and console output are all
gated to rank zero, and DDP wrapping plus `no_sync()` on non-final accumulation
micro-steps are handled for you:

```bash
uv run torchrun --nproc_per_node=4 -m dlt.train experiment=mine
```

Single-device is the same code with `world_size == 1`. Add a `DistributedSampler` in
your DataModule when `world_size > 1`.

**Not verified on hardware** — this project's development machine has no compatible GPU,
so the distributed and `torch.compile` paths are written but untested. Treat them as
first things to check on a real multi-GPU box. Note also that DDP handles
double-backward poorly, so models with `create_graph=True` physics losses are better run
single-device.

---

## Troubleshooting

**`CUBLAS_STATUS_ARCH_MISMATCH`** — `torch.cuda.is_available()` returned True for a GPU
your torch build has no kernels for. Install a matching torch build, or pass
`trainer.device=cpu`.

**`ValueError: badly formed help string` on startup** — you're on Python 3.14. hydra-core
1.3.4 is incompatible; use 3.13 (`uv python pin 3.13 && uv sync`).

**`amp=... requires dtype='float32'`** — working as intended; see
[Precision](#precision-dtype-and-amp).

**Resume seems to do nothing** — you resumed at a step already at or past `max_steps`.
Raise `trainer.max_steps`.

**Cold optimizer after resume** — should not happen; resume is handled inside `fit()`.
If you call `load_checkpoint` yourself *before* `fit()`, the optimizers you populate are
replaced by the ones `fit()` builds. Pass `resume=` to `fit()` instead.

**`FileNotFoundError: checkpoint directory not found`** — `resume` and `ckpt` take a
*directory* (`.../ckpt/best`), not a file.

**Every run's log lands in the repo root** — `hydra.job_logging`'s file handler must name
`${hydra.runtime.output_dir}` explicitly, because `hydra.job.chdir` is false.

**`HydraConfig was not set` during `--multirun`** — `hydra.sweep.dir` is resolved before
`HydraConfig` exists, so nothing it references may use a `${hydra:...}` resolver. Keep
`paths.output_root` a plain relative string.
