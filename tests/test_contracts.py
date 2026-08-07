"""Auto-parametrized over EVERY model config. This is the "no boilerplate" claim.

Drop in configs/model/yours.yaml and it is automatically checked that it
instantiates, its step runs, the loss is finite and scalar, backward populates
gradients, and configure_optimizers returns something valid. You write no test code.

Each model config is paired with its datamodule by MODEL_DATA below -- the one line
you add per new model.
"""

from __future__ import annotations

import hydra
import pytest
import torch

from engine.base import OptimSpec, TaskModule, TrainState
from engine.utils import move_to_device
from tests.conftest import config_names, load

# model config -> data config it should be tested against.
MODEL_DATA = {"mlp": "synthetic", "forecast": "series", "pinn": "collocation"}

MODELS = config_names("model")


def _build(name: str):
    cfg = load([f"model={name}", f"data={MODEL_DATA[name]}"])
    module = hydra.utils.instantiate(cfg.model)
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup("fit")
    return module, datamodule


def _first_batch(datamodule):
    loader = datamodule.train_dataloader()
    loader = next(iter(loader.values())) if isinstance(loader, dict) else loader
    return next(iter(loader))


@pytest.mark.parametrize("name", MODELS)
def test_model_config_has_data_pairing(name: str) -> None:
    assert name in MODEL_DATA, (
        f"add {name!r} to MODEL_DATA in tests/test_contracts.py so it gets tested"
    )


@pytest.mark.parametrize("name", MODELS)
def test_is_a_taskmodule(name: str) -> None:
    module, _ = _build(name)
    assert isinstance(module, TaskModule)


@pytest.mark.parametrize("name", MODELS)
def test_training_step_returns_finite_scalar_loss(name: str) -> None:
    module, datamodule = _build(name)
    out = module.training_step(_first_batch(datamodule), TrainState())
    assert "loss" in out, "training_step must return a 'loss' entry"
    loss = out["loss"]
    assert loss.ndim == 0, f"loss must be a scalar, got shape {tuple(loss.shape)}"
    assert torch.isfinite(loss), "loss is not finite at init"


@pytest.mark.parametrize("name", MODELS)
def test_backward_populates_gradients(name: str) -> None:
    module, datamodule = _build(name)
    out = module.training_step(_first_batch(datamodule), TrainState())
    out["loss"].backward()
    trainable = [p for p in module.parameters() if p.requires_grad]
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable), (
        "backward() populated no finite gradients on any trainable parameter"
    )


@pytest.mark.parametrize("name", MODELS)
def test_validation_step_runs(name: str) -> None:
    module, datamodule = _build(name)
    loaders = datamodule.val_dataloader()
    loaders = loaders if isinstance(loaders, dict) else {"": loaders}
    for loader in loaders.values():
        batch = next(iter(loader))
        # eval_requires_grad models (physics losses) must not be run under no_grad.
        ctx = torch.enable_grad() if module.eval_requires_grad else torch.no_grad()
        with ctx:
            out = module.validation_step(batch, TrainState())
        assert out and all(torch.isfinite(v) for v in out.values() if torch.is_tensor(v))


@pytest.mark.parametrize("name", MODELS)
def test_configure_optimizers(name: str) -> None:
    module, _ = _build(name)
    spec = module.configure_optimizers()
    assert isinstance(spec, OptimSpec)
    assert spec.optimizers, "at least one optimizer required"
    for s in spec.schedulers:
        assert s.interval in {"step", "epoch"}


@pytest.mark.parametrize("name", MODELS)
def test_checkpoint_roundtrip(name: str, tmp_path) -> None:
    """Weights AND buffers must survive. Scaler statistics live in buffers precisely
    so they cannot desync from the weights across a save/load."""
    module, datamodule = _build(name)
    spec = module.configure_optimizers()
    from engine.checkpoint import load_checkpoint, save_checkpoint

    batch = _first_batch(datamodule)
    ctx = torch.enable_grad() if module.eval_requires_grad else torch.no_grad()
    with ctx:
        before = module.validation_step(batch, TrainState())["loss"].item()

    save_checkpoint(tmp_path / "ck", module, optimizers=spec.optimizers, state=TrainState())

    fresh, _ = _build(name)
    load_checkpoint(tmp_path / "ck", fresh)
    with torch.enable_grad() if fresh.eval_requires_grad else torch.no_grad():
        after = fresh.validation_step(batch, TrainState())["loss"].item()

    assert before == pytest.approx(after, rel=1e-5), "checkpoint round-trip changed predictions"


def test_batch_is_never_inspected_by_device_move() -> None:
    """Duck-typed movement is what lets graph batches and custom containers work
    without the framework knowing what they are."""

    class Custom:
        def __init__(self):
            self.moved = False

        def to(self, device):
            self.moved = True
            return self

    dev = torch.device("cpu")
    obj = move_to_device({"a": [torch.ones(2)], "b": Custom(), "c": "untouched"}, dev)
    assert obj["b"].moved
    assert obj["c"] == "untouched"
