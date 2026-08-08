"""End-to-end: the loop, the tracking, and the claims that are actually falsifiable."""

from __future__ import annotations

import json

import hydra
import pytest
import torch

from engine.base import TrainState
from engine.tracking import JSONLLogger, MultiLogger, RunMeta, artifacts_dir
from engine.trainer import Trainer
from engine.utils import MetricAccumulator, ScheduledValue
from tests.conftest import load


def _fit(overrides, **trainer_kw):
    cfg = load(overrides)
    module = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.data)
    kw = {"max_steps": 20, "val_every": 10, "log_every": 5, "device": "cpu"}
    kw.update(trainer_kw)
    trainer = Trainer(**kw)
    result = trainer.fit(module, datamodule)
    return trainer, result


def test_two_step_train_runs() -> None:
    trainer, result = _fit(["experiment=e0"], max_steps=2, val_every=2, log_every=1)
    assert trainer.state.global_step == 2
    assert result is not None and torch.isfinite(torch.tensor(result))


def test_global_step_counts_optimizer_steps_not_microbatches() -> None:
    """With grad_accum=4, 5 optimizer steps means 20 forward/backward passes. Counting
    micro-batches instead would put every schedule and logged x-axis off by 4x."""
    trainer, _ = _fit(["experiment=e0"], max_steps=5, grad_accum=4, val_every=0, log_every=0)
    assert trainer.state.global_step == 5
    assert trainer.state.samples_seen == 5 * 4 * 32  # steps x accum x batch_size


def test_multi_horizon_val_namespaces_metrics() -> None:
    """dict-of-loaders -> val/h8/..., val/h24/... The same mechanism serves
    multi-resolution evaluation."""
    trainer, _ = _fit(
        ["experiment=forecast"], max_steps=4, val_every=4, log_every=0, monitor="val/h8/loss"
    )
    keys = set(trainer.state.metrics)
    assert any(k.startswith("val/h8/") for k in keys), keys
    assert any(k.startswith("val/h24/") for k in keys), keys


def test_rollout_runs_on_its_own_cadence() -> None:
    trainer, _ = _fit(
        ["experiment=forecast"],
        max_steps=4,
        val_every=0,
        rollout_every=4,
        log_every=0,
        monitor="val/h8/loss",
    )
    assert any(k.startswith("rollout/") for k in trainer.state.metrics)


def test_pinn_converges_toward_the_analytic_solution() -> None:
    """The point of choosing du/dx = -u: the exact solution is e^(-x), so this is a
    real assertion rather than 'the loss went down'."""
    torch.manual_seed(0)
    trainer, _ = _fit(
        ["experiment=pinn"], max_steps=600, val_every=600, log_every=0, monitor="val/l2_error"
    )
    err = trainer.state.metrics["val/l2_error"]
    assert err < 0.05, f"PINN did not approach e^(-x): mean |u - e^-x| = {err}"


def test_eval_is_not_wrapped_in_no_grad_for_physics_models() -> None:
    """A blanket torch.no_grad() in eval makes PDE residuals uncomputable."""
    cfg = load(["experiment=pinn"])
    module = hydra.utils.instantiate(cfg.model)
    assert module.eval_requires_grad is True
    trainer = Trainer(device="cpu")
    trainer.module = module
    assert not isinstance(trainer._grad_ctx(), torch.no_grad)


def test_manual_optimization_skips_the_trainer_update() -> None:
    """The escape hatch is a ramp: the Trainer stays out of the way but its services
    (backward, clip_and_step) remain callable."""
    cfg = load(["experiment=e0"])
    module = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.data)

    calls = []
    original = module.training_step

    def manual_step(batch, state):
        out = original(batch, state)
        module.trainer.backward(out["loss"])
        module.trainer.clip_and_step(module.trainer.optimizers[0])
        calls.append(state.global_step)
        return {"loss": out["loss"].detach()}

    module.manual_optimization = True
    module.training_step = manual_step

    trainer = Trainer(max_steps=3, val_every=0, log_every=0, device="cpu")
    trainer.fit(module, datamodule)
    assert len(calls) == 3


def test_metric_accumulator_weights_by_batch_size() -> None:
    """Mean-of-means is wrong for variable-size batches."""
    acc = MetricAccumulator()
    acc.update({"x": 1.0}, weight=1.0)
    acc.update({"x": 3.0}, weight=3.0)
    assert acc.compute()["x"] == pytest.approx((1 * 1 + 3 * 3) / 4)


def test_metric_accumulator_drops_non_finite() -> None:
    acc = MetricAccumulator()
    acc.update({"x": 1.0})
    acc.update({"x": float("nan")})
    assert acc.compute()["x"] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "schedule", ["constant", "linear", "cosine", "exponential", "inverse_sigmoid"]
)
def test_scheduled_value_endpoints(schedule: str) -> None:
    s = ScheduledValue(start=1.0, end=0.1, over_steps=100, schedule=schedule)
    assert s(0) == pytest.approx(1.0, abs=1e-6)
    if schedule != "constant":
        assert s(100) == pytest.approx(0.1, abs=1e-6)
        assert s(10_000) == pytest.approx(0.1, abs=1e-6)  # clamped past the end


def test_jsonl_and_run_meta_are_written(tmp_path) -> None:
    logger = MultiLogger([JSONLLogger(tmp_path / "metrics.jsonl")])
    logger.log_scalars({"train/loss": 0.5}, 1)
    logger.close()
    rec = json.loads((tmp_path / "metrics.jsonl").read_text().splitlines()[0])
    assert rec == {"step": 1, "train/loss": 0.5}

    meta = RunMeta(tmp_path, "exp0", "abc123")
    assert json.loads((tmp_path / "run_meta.json").read_text())["status"] == "running"
    meta.finish("finished", {"val/loss": 0.25})
    done = json.loads((tmp_path / "run_meta.json").read_text())
    assert done["status"] == "finished" and done["metrics"]["val/loss"] == 0.25


def test_artifacts_dir_is_usable_without_hydra(tmp_path) -> None:
    d = artifacts_dir(tmp_path, "rollout_h1000")
    assert d.is_dir() and d == tmp_path / "artifacts" / "rollout_h1000"


def test_a_broken_logger_cannot_kill_a_run() -> None:
    class Exploding(JSONLLogger):
        def __init__(self):
            pass

        def log_scalars(self, metrics, step):
            raise RuntimeError("boom")

        def close(self):
            pass

    MultiLogger([Exploding()]).log_scalars({"a": 1.0}, 0)  # must not raise


def test_temporal_split_does_not_leak_future_into_past() -> None:
    """A shuffled split makes the val loss lie."""
    cfg = load(["experiment=forecast"])
    dm = hydra.utils.instantiate(cfg.data)
    dm.setup("fit")
    full = torch.cat([dm.train_series, dm.val_series])
    assert torch.equal(full[: len(dm.train_series)], dm.train_series)
    assert len(dm.val_series) > 0


def test_scaler_stats_are_buffers_and_ride_the_checkpoint() -> None:
    cfg = load(["experiment=forecast"])
    module = hydra.utils.instantiate(cfg.model)
    assert "mean" in dict(module.named_buffers())
    assert "std" in dict(module.named_buffers())
    module.set_scaler(3.0, 2.0)
    assert module.state_dict()["mean"].item() == pytest.approx(3.0)


def test_on_data_ready_hands_the_datamodule_scaler_to_the_model() -> None:
    """Regression: SeriesData fitted a scaler that nothing ever read, so the model
    normalised with mean=0/std=1 and the fitted statistics were dead code."""
    cfg = load(["experiment=forecast"])
    module = hydra.utils.instantiate(cfg.model)
    dm = hydra.utils.instantiate(cfg.data)
    assert (module.mean.item(), module.std.item()) == (0.0, 1.0), "precondition"

    Trainer(max_steps=1, val_every=0, log_every=0, device="cpu").fit(module, dm)

    assert module.mean.item() == pytest.approx(dm.scaler[0], rel=1e-5)
    assert module.std.item() == pytest.approx(dm.scaler[1], rel=1e-5)
    assert module.std.item() != 1.0, "the scaler never reached the model"


def test_checkpointed_stats_beat_freshly_computed_ones_on_resume(tmp_path) -> None:
    """Ordering claim in fit(): on_data_ready runs BEFORE the checkpoint loads, so a
    resumed run keeps the statistics it trained with rather than today's data's."""
    from engine.checkpoint import save_checkpoint

    cfg = load(["experiment=forecast"])
    saved = hydra.utils.instantiate(cfg.model)
    saved.set_scaler(3.0, 2.0)
    save_checkpoint(tmp_path / "ck", saved, state=TrainState())

    fresh = hydra.utils.instantiate(cfg.model)
    dm = hydra.utils.instantiate(cfg.data)
    Trainer(max_steps=1, val_every=0, log_every=0, device="cpu").fit(
        fresh, dm, resume=tmp_path / "ck"
    )

    assert fresh.mean.item() == pytest.approx(3.0), "the datamodule overwrote the checkpoint"


def test_step_snapshots_are_step_tagged_and_pruned(tmp_path) -> None:
    """every_steps keeps permanent snapshots; keep_last bounds the disk they use."""
    from engine.callbacks import Checkpoint

    ckpt = tmp_path / "ckpt"
    cb = Checkpoint(
        dirpath=ckpt,
        every_steps=2,
        keep_last=2,
        last_every=0,
        save_best=False,
        save_best_weights=False,
    )
    _fit(["experiment=e0"], max_steps=10, val_every=0, log_every=0, callbacks=[cb])

    # steps 2,4,6,8,10 were written; only the newest two survive.
    assert sorted(p.name for p in ckpt.glob("step_*")) == ["step_00000008", "step_00000010"]
    # Zero padding is load-bearing: sorted() must agree with numeric order past step 10.
    assert sorted(["step_00000009", "step_00000010"]) == ["step_00000009", "step_00000010"]


def test_best_weights_carries_no_optimizer_state_and_still_loads(tmp_path) -> None:
    """The inference artifact. Optimizer moments are ~2x the weights and are exactly
    what you do not want to ship."""
    from engine.callbacks import Checkpoint
    from engine.checkpoint import load_checkpoint

    ckpt = tmp_path / "ckpt"
    cb = Checkpoint(dirpath=ckpt, every_steps=0, last_every=0, save_last=False)
    _fit(["experiment=e0"], max_steps=4, val_every=2, log_every=0, callbacks=[cb])

    assert (ckpt / "best" / "state.pt").exists(), "best/ must stay resumable"
    assert (ckpt / "best_weights" / "model.safetensors").exists()
    assert not (ckpt / "best_weights" / "state.pt").exists()

    fresh = hydra.utils.instantiate(load(["experiment=e0"]).model)
    load_checkpoint(ckpt / "best_weights", fresh)  # no state.pt -> must not raise


def test_resume_restores_step_and_optimizer_state(tmp_path) -> None:
    """Regression: resume used to load into optimizers that fit() then replaced, so a
    resumed run silently continued with a cold optimizer -- no error, just different
    training. Adam momentum must survive, not just the weights and the step counter.
    """
    from engine.checkpoint import save_checkpoint

    trainer, _ = _fit(["experiment=e0"], max_steps=10, val_every=0, log_every=0)
    save_checkpoint(
        tmp_path / "ck", trainer.raw, optimizers=trainer.optimizers, state=trainer.state
    )
    saved_moments = len(trainer.optimizers[0].state_dict()["state"])
    assert saved_moments > 0, "precondition: the optimizer should have accumulated state"

    cfg = load(["experiment=e0"])
    fresh = hydra.utils.instantiate(cfg.model)
    dm = hydra.utils.instantiate(cfg.data)
    resumed = Trainer(max_steps=10, val_every=0, log_every=0, device="cpu")
    resumed.fit(fresh, dm, resume=tmp_path / "ck")

    assert resumed.state.global_step == 10, "step counter did not resume"
    assert len(resumed.optimizers[0].state_dict()["state"]) == saved_moments, (
        "optimizer state was discarded on resume"
    )


def test_grad_stats_sees_live_gradients(tmp_path) -> None:
    """Regression: GradStats ran on on_train_batch_end, which fires after
    clip_and_step's zero_grad(set_to_none=True). Every .grad was None, so it logged a
    global norm of 0.0 and zero histograms -- a diagnostic that silently reports no
    gradient problems forever.
    """
    from engine.callbacks import GradStats

    class Recorder:
        def __init__(self):
            self.scalars, self.histograms = [], []

        def log_scalars(self, metrics, step):
            self.scalars.append(metrics)

        def log_histogram(self, tag, values, step):
            self.histograms.append(tag)

    rec = Recorder()
    cfg = load(["experiment=e0"])
    trainer = Trainer(
        max_steps=4,
        val_every=0,
        log_every=0,
        device="cpu",
        callbacks=[GradStats(every=1)],
        logger=rec,
    )
    trainer.fit(hydra.utils.instantiate(cfg.model), hydra.utils.instantiate(cfg.data))

    norms = [d["grad/global_norm"] for d in rec.scalars if "grad/global_norm" in d]
    assert norms, "GradStats logged no global norm at all"
    assert all(n > 0 for n in norms), f"gradients were already cleared: {norms}"
    assert rec.histograms, "GradStats logged no histograms"


def test_early_stopping_fires_when_metric_stops_improving() -> None:
    from engine.callbacks import EarlyStopping

    trainer, _ = _fit(
        ["experiment=e0"],
        max_steps=100,
        val_every=5,
        log_every=0,
        monitor="val/loss",
        monitor_mode="max",  # loss decreases, so "max" never improves
        callbacks=[EarlyStopping(patience=2)],
    )
    assert trainer.state.should_stop
    assert trainer.state.global_step < 100, "early stopping did not cut the run short"


def test_save_extra_load_extra_roundtrip(tmp_path) -> None:
    """Non-tensor state (tokenizers, vocabs) -- safetensors holds tensors only."""
    import json

    from engine.checkpoint import load_checkpoint, save_checkpoint

    cfg = load(["experiment=e0"])
    module = hydra.utils.instantiate(cfg.model)
    module.vocab = {"a": 1, "b": 2}
    module.save_extra = lambda d: (d / "vocab.json").write_text(json.dumps(module.vocab))
    save_checkpoint(tmp_path / "ck", module)
    assert (tmp_path / "ck" / "extra" / "vocab.json").exists()

    fresh = hydra.utils.instantiate(cfg.model)
    seen = {}
    fresh.load_extra = lambda d: seen.update(json.loads((d / "vocab.json").read_text()))
    load_checkpoint(tmp_path / "ck", fresh)
    assert seen == {"a": 1, "b": 2}


def test_datamodule_state_is_round_tripped(tmp_path) -> None:
    """If a DataModule defines state_dict/load_state_dict, resume restores data position."""
    from engine.checkpoint import load_checkpoint, save_checkpoint

    cfg = load(["experiment=e0"])
    base = hydra.utils.instantiate(cfg.data)

    class Stateful(type(base)):
        restored = None

        def state_dict(self):
            return {"consumed": 4242}

        def load_state_dict(self, sd):
            self.restored = sd.get("consumed")

    dm = Stateful()
    dm.setup("fit")
    module = hydra.utils.instantiate(cfg.model)
    save_checkpoint(tmp_path / "ck", module, datamodule=dm)

    fresh = Stateful()
    fresh.setup("fit")
    load_checkpoint(tmp_path / "ck", hydra.utils.instantiate(cfg.model), datamodule=fresh)
    assert fresh.restored == 4242


def _ddp_worker(rank: int, world: int, port: int, out: str) -> None:
    """One rank of the DDP regression test. Module level so spawn can pickle it."""
    import os

    os.environ.update(
        RANK=str(rank),
        WORLD_SIZE=str(world),
        LOCAL_RANK=str(rank),
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
    )
    from dataset.examples import SyntheticData
    from engine.trainer import Trainer
    from models.mlp import MLP

    torch.manual_seed(0)  # identical init; only the DATA differs per rank
    module = MLP()
    dm = SyntheticData(n_train=128, n_val=32, batch_size=16, seed=100 + rank)
    Trainer(max_steps=3, val_every=0, log_every=0, device="cpu").fit(module, dm)
    torch.save(torch.cat([p.detach().flatten() for p in module.parameters()]), f"{out}/{rank}.pt")


def test_ddp_ranks_actually_all_reduce(tmp_path) -> None:
    """Regression: the Trainer wrapped the module in DDP and then called
    self.raw.training_step, walking straight past DistributedDataParallel.forward. The
    reducer's prepare_for_backward never ran, so NOTHING was ever all-reduced and the
    ranks silently trained into different models -- no error, and a healthy loss curve
    on every rank. no_sync() was decorative.

    Feeding each rank different data is what makes this falsifiable: identical data
    gives identical gradients and the test passes either way.
    """
    import socket

    import torch.multiprocessing as mp

    with socket.socket() as s:
        s.bind(("", 0))
        port = s.getsockname()[1]

    mp.spawn(_ddp_worker, args=(2, port, str(tmp_path)), nprocs=2, join=True)
    a, b = torch.load(tmp_path / "0.pt"), torch.load(tmp_path / "1.pt")
    assert torch.equal(a, b), f"ranks diverged: max |a-b| = {(a - b).abs().max().item()}"


def test_complex_weights_are_split_in_histograms_and_named_in_checkpoint_errors(
    tmp_path,
) -> None:
    """The complex path was reasoned about but never executed: no shipped model has
    complex weights, so neither the histogram split nor the dtype error had ever run.

    add_histogram does NOT reject complex -- it casts to real and silently drops the
    imaginary part -- so a single tag here would mean diagnostics that quietly tell
    half the truth. safetensors 0.8.0 round-trips complex64 but raises KeyError on
    complex128, which is why the error has to name the escape hatch.
    """
    from engine.base import OptimSpec, TaskModule
    from engine.callbacks import GradStats
    from engine.checkpoint import SafetensorsFormat
    from engine.tracking import TensorBoardLogger

    class Spectral(TaskModule):
        def __init__(self, dtype=torch.complex64):
            super().__init__()
            self.w = torch.nn.Parameter(torch.randn(8, 1, dtype=dtype))

        def training_step(self, batch, state):
            return {"loss": (batch["x"].to(self.w.dtype) @ self.w).abs().pow(2).mean()}

        validation_step = training_step

        def configure_optimizers(self):
            return OptimSpec.of(torch.optim.Adam(self.parameters(), lr=1e-3))

    class RecordingWriter:
        def __init__(self):
            self.tags = []

        def add_histogram(self, tag, values, step):
            assert not values.is_complex(), f"{tag} reached add_histogram still complex"
            self.tags.append(tag)

        def add_scalar(self, *a, **kw): ...
        def flush(self): ...
        def close(self): ...

    logger = TensorBoardLogger(tmp_path / "tb")
    logger.writer = RecordingWriter()

    cfg = load(["experiment=e0"])
    trainer = Trainer(
        max_steps=2,
        val_every=0,
        log_every=0,
        device="cpu",
        callbacks=[GradStats(every=1)],
        logger=logger,
    )
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trainer.fit(Spectral(), hydra.utils.instantiate(cfg.data))

    # Nothing on this path may quietly drop the imaginary half -- not the dtype cast in
    # fit(), not the grad norm in GradStats, not the histograms.
    dropped = [str(w.message) for w in caught if "imaginary" in str(w.message)]
    assert not dropped, dropped

    # fit() casts the module to param_dtype, and Module.to(float32) converts complex
    # parameters too -- which would drop the imaginary part before training even began.
    assert trainer.raw.w.is_complex(), "fit() flattened a complex parameter to real"

    tags = set(logger.writer.tags)
    assert {"weights/w/abs", "weights/w/real", "weights/w/imag"} <= tags, tags
    assert {"grads/w/abs", "grads/w/real", "grads/w/imag"} <= tags, tags
    assert "weights/w" not in tags, "complex tensor was logged as a single real histogram"

    SafetensorsFormat().save(trainer.raw, tmp_path / "c64.safetensors")
    restored = Spectral()
    SafetensorsFormat().load(restored, tmp_path / "c64.safetensors")
    assert torch.equal(restored.w.detach(), trainer.raw.w.detach())

    with pytest.raises(RuntimeError, match="checkpoint.format"):
        SafetensorsFormat().save(Spectral(torch.complex128), tmp_path / "c128.safetensors")


def test_trainstate_roundtrips() -> None:
    s = TrainState(global_step=7, epoch=2, samples_seen=99, metrics={"a": 1.0})
    t = TrainState()
    t.load_state_dict(s.state_dict())
    assert (t.global_step, t.epoch, t.samples_seen) == (7, 2, 99)
