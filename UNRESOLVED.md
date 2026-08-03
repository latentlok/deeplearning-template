# Unresolved

Handoff notes. Everything here is either **untested** (not known-broken) or a
**deliberate omission** — the distinction matters, so don't "fix" the second group.

State as of commit `b45734a`: 61 tests pass, lint clean, three examples run end to end,
fresh clone reproduces bit-identically. Read `CLAUDE.md` first, then `ARCHITECTURE.md`.

## Getting running on a new machine

```bash
git clone https://github.com/latentlok/deeplearning-template.git
cd deeplearning-template
uv sync --extra dev          # .python-version pins 3.13; uv fetches it if needed
uv run pytest tests/ -q      # expect 61 passed
uv run python -m dlt.train experiment=e0
```

If `uv sync` resolves to Python 3.14 you will get `ValueError: badly formed help string`
from Hydra — see item 6.

---

## 1. GPU, DDP and `torch.compile` are written but never executed

**Highest-value thing to close.** All verification so far is CPU-only: this project's
development machine has a GTX 1050 (sm_61) and torch 2.13 ships CUDA kernels starting at
sm_75, so `torch.cuda.is_available()` returns True but any matmul dies with
`CUBLAS_STATUS_ARCH_MISMATCH`.

On real hardware, run and report:

```bash
uv run python -m dlt.train experiment=forecast                    # device=auto -> cuda
uv run python -m dlt.train experiment=e0 amp=bf16                 # autocast + GradScaler
uv run python -m dlt.train experiment=e0 trainer.compile=true     # torch.compile
uv run torchrun --nproc_per_node=2 -m dlt.train experiment=e0     # DDP
```

Specific things to check, because they are where the untested code is:

- **DDP**: exactly one TensorBoard event file, one `ckpt/` and one `run_meta.json` — no
  duplicated console output. Rank-zero gating is in `core/utils.is_rank_zero`.
- **DDP + `grad_accum > 1`**: `no_sync()` on non-final micro-steps (`core/trainer.py`,
  `train_step`). Verify loss curves match single-device at the same effective batch.
- **`amp=fp16`**: `GradScaler` is enabled only for fp16, not bf16. Check for inf/NaN.
- **PINN under DDP**: expected to be poor — DDP handles double-backward badly. Prefer
  single-device for `create_graph=True` models. Confirm rather than assume.

Note `trainer.device=auto` deliberately does **not** probe architecture compatibility —
that was explicitly descoped. On a mismatched box, pass `trainer.device=cpu`.

## 2. Complex dtypes are reasoned about, not exercised

`TensorBoardLogger.log_histogram` splits complex tensors into abs/real/imag because
`add_histogram` silently discards the imaginary part (measured — it emits only a
`ComplexWarning`). But **no shipped model has complex weights**, so the split path has
never run.

Also measured: safetensors 0.8.0 round-trips `complex64` but raises `KeyError` on
`complex128`. Since complex follows the real dtype, a model at `dtype: float64` with
spectral weights cannot use the default checkpoint format.

To close: add a small model with a `complex64` parameter (an FNO-style spectral layer is
the natural one), enable `grad_stats`, and confirm three histograms appear per complex
tensor. Then try `dtype=float64` and confirm the checkpoint error names
`checkpoint.format=torch` as the fix.

## 3. `notebooks/explore_run.ipynb` was planned and never written

Deliberately skipped — a stub notebook is noise, and `scripts/runs.py` plus
`metrics.jsonl` cover the actual need. Write one only if a real workflow wants it.

## 4. Weights-only fine-tuning has no CLI flag

By decision, not oversight. `resume=<dir>` does an exact resume; fine-tuning from
someone else's weights is `load_checkpoint(..., weights_only=True)`, documented in
`USAGE.md` as an `init_from` argument on your model so it stays configurable and still
runs through `dlt.train`. Don't add a flag unless asked.

## 5. No DataModule implements `state_dict()` / `load_state_dict()`

The mechanism is tested (`test_datamodule_state_is_round_tripped`), but no shipped
example uses it, so resume restores model and optimizer while the dataloader restarts
from the top. The Trainer logs a warning saying exactly that.

This only bites for streaming/`IterableDataset` training, where it means silently
retraining the same prefix. Implement the pair on your DataModule when you get there —
recipe is in `USAGE.md` under *Checkpoints and resuming*.

## 6. Python is pinned to 3.13 because hydra-core 1.3.4 breaks on 3.14

Python 3.14's argparse validates help strings and rejects Hydra's lazy
`--shell-completion` help object, so `@hydra.main` dies before any project code runs.
`requires-python = ">=3.12,<3.14"` records this.

`hydra-core 1.4.0.dev6` declares 3.10–3.14 and needs `omegaconf>=2.4.0.dev13`. When 1.4
ships **stable**, lift the ceiling and re-run the suite. Not worth two pre-release
dependencies before then.

## 7. The committed code graph goes stale

`graphify-out/graph.json` (465 nodes, 1092 edges) is a build artifact committed for
convenience. Refresh after any code change:

```bash
graphify update . --no-cluster        # ~2s, LLM-free, nothing leaves the machine
```

Semantic clustering was **deliberately not run**. Never invoke `graphify extract`,
`label` or `cluster-only` without an explicit `--backend`: bare, graphify picks one from
the environment (AWS Bedrock if `AWS_PROFILE` is set), which bills a real account and
ships the code off-box.

## 8. Smaller known edges

- `scripts/runs.py --where` matches the override string **literally**, so
  `model.lr=0.005` will not match a run launched as `model.lr=5e-3`.
- `EarlyStopping` and `GradStats` are implemented and tested but commented out in
  `configs/callbacks/default.yaml` — enable per experiment.
- `data/raw`, `data/processed`, `data/cache` are gitignored, so a fresh clone has no
  `data/` tree. Nothing shipped needs it (all three examples are synthetic).

---

## Deliberately out of scope — do not add without being asked

- **HPO / sweeper.** No Optuna, no plugin. `train.py` returns the monitored metric,
  which is the whole coupling surface one needs; adding it later touches nothing in
  `src/`.
- **FSDP2 / model parallelism.** DDP only.
- **Architecture-compatibility probing in `device: auto`.** Explicitly descoped.
- **Model-specific machinery in `core/`** (FNO, GNO, Transolver…). The whole design is
  fork-per-model-family; `core/` stays general and the fork adds what it needs. Extension
  points: subclass `Trainer`, `Logger`, `WeightFormat`, or `Callback`.

## Two bugs already found and fixed — worth knowing the pattern

Both were invisible to code review and obvious on execution, which is the working style
`CLAUDE.md` asks for:

1. **Resume discarded optimizer state** (`1b9fc9b`) — the checkpoint loaded into
   optimizers that `fit()` then replaced, so a resumed run silently continued with a cold
   optimizer.
2. **`GradStats` never saw gradients** (`b45734a`) — it ran on `on_train_batch_end`, which
   fires after `zero_grad(set_to_none=True)`, so it reported a global norm of 0.0 forever.

Both now have regression tests. When a test passes, check it passed for the *right
reason* — an earlier resume test passed only because the run exited before touching the
optimizer.
