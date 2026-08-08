"""The two files a fork actually rewrites: dataset/loader.py and utils/stats.py.

The store is built in tmp_path, so nothing here needs a mounted dataset.

The chain under test, end to end:

    train.zarr -> stats.json -> ZarrData.stats -> model buffers -> checkpoint

A break anywhere in it is silent: the model still trains, and inference normalises
against different numbers than training did.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import zarr

from dataset.loader import ZarrData, ZarrRows, open_group
from engine.trainer import Trainer
from models.mlp import MLP
from utils.stats import compute_stats

INPUTS, TARGETS = ["feature_1", "feature_2"], ["feature_3"]


def _write_store(path, n: int, seed: int = 0) -> dict[str, np.ndarray]:
    """time + three features. feature_3 is a real function of the other two, so a model
    that learns nothing is distinguishable from one that does. The features are given
    very different scales on purpose -- that is what normalisation has to fix."""
    rng = np.random.default_rng(seed)
    f1 = rng.normal(loc=100.0, scale=5.0, size=n).astype("float32")
    f2 = rng.normal(loc=-0.01, scale=0.002, size=n).astype("float32")
    variables = {
        "time": np.arange(n, dtype="float64"),
        "feature_1": f1,
        "feature_2": f2,
        "feature_3": (0.7 * f1 + 300.0 * f2).astype("float32"),
    }
    group = zarr.open_group(str(path), mode="w")
    for name, values in variables.items():
        group.create_array(name, shape=values.shape, dtype=values.dtype, chunks=(16,))[:] = values
    return variables


def _make_root(tmp_path, n_train: int = 64, n_val: int = 16):
    _write_store(tmp_path / "train.zarr", n_train, seed=0)
    _write_store(tmp_path / "val.zarr", n_val, seed=1)
    return tmp_path


def _write_stats(root) -> dict:
    group = open_group(root / "train.zarr")
    stats = {name: compute_stats(group[name]) for name in sorted(group.array_keys())}
    (root / "stats.json").write_text(json.dumps(stats))
    return stats


# -- what one sample is ------------------------------------------------------------


def test_a_sample_is_the_named_variables_at_that_row(tmp_path) -> None:
    variables = _write_store(tmp_path / "train.zarr", 32)
    ds = ZarrRows(open_group(tmp_path / "train.zarr"), INPUTS, TARGETS)

    assert len(ds) == 32
    assert ds[7]["x"].tolist() == pytest.approx(
        [variables["feature_1"][7], variables["feature_2"][7]]
    )
    assert ds[7]["y"].tolist() == pytest.approx([variables["feature_3"][7]])


def test_unnamed_variables_are_never_read(tmp_path) -> None:
    """`time` is in the store. Nothing selected it, so it must not reach the batch."""
    variables = _write_store(tmp_path / "train.zarr", 32)
    ds = ZarrRows(open_group(tmp_path / "train.zarr"), INPUTS, TARGETS)

    assert ds[7]["x"].shape == (2,), "width must be len(inputs), not every variable"
    assert variables["time"][7] not in ds[7]["x"].tolist()


def test_input_order_follows_the_config(tmp_path) -> None:
    """Reordering `inputs` reorders the vector -- otherwise in_dim stays right and the
    columns silently swap."""
    _write_store(tmp_path / "train.zarr", 32)
    group = open_group(tmp_path / "train.zarr")
    forward = ZarrRows(group, ["feature_1", "feature_2"], TARGETS)[3]["x"].tolist()
    reverse = ZarrRows(group, ["feature_2", "feature_1"], TARGETS)[3]["x"].tolist()
    assert forward == reverse[::-1]


# -- errors that would otherwise be confusing ---------------------------------------


def test_missing_variable_lists_what_the_store_has(tmp_path) -> None:
    _write_store(tmp_path / "train.zarr", 8)
    with pytest.raises(KeyError, match="feature_9"):
        ZarrRows(open_group(tmp_path / "train.zarr"), ["feature_9"], TARGETS)


def test_pointing_at_an_array_instead_of_a_group_says_so(tmp_path) -> None:
    """The common mistake; zarr's own ContainsArrayError never mentions groups."""
    _write_store(tmp_path / "train.zarr", 8)
    with pytest.raises(ValueError, match="not a group"):
        open_group(tmp_path / "train.zarr" / "feature_1")


def test_missing_store_points_at_the_data_root(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="DL_DATA"):
        ZarrData(root=tmp_path / "nowhere").setup("fit")


def test_variables_of_different_lengths_are_rejected(tmp_path) -> None:
    """Two variables written by different runs. Silent otherwise -- the shorter one
    just indexes past its end partway through an epoch."""
    group = zarr.open_group(str(tmp_path / "train.zarr"), mode="w")
    for name, n in (("feature_1", 10), ("feature_2", 7), ("feature_3", 10)):
        group.create_array(name, shape=(n,), dtype="float32", chunks=(8,))
    with pytest.raises(ValueError, match="disagree on length"):
        ZarrRows(open_group(tmp_path / "train.zarr"), INPUTS, TARGETS)


# -- statistics ---------------------------------------------------------------------


def test_stats_match_numpy(tmp_path) -> None:
    """Streaming accumulation must agree with the obvious one-shot computation."""
    variables = _write_store(tmp_path / "train.zarr", 100)
    group = open_group(tmp_path / "train.zarr")
    stats = compute_stats(group["feature_1"], chunk=7)  # not a divisor of 100

    f1 = variables["feature_1"].astype("float64")
    assert stats["count"] == 100
    assert stats["mean"] == pytest.approx(f1.mean(), rel=1e-6)
    assert stats["std"] == pytest.approx(f1.std(), rel=1e-6)
    assert stats["min"] == pytest.approx(f1.min(), rel=1e-6)
    assert stats["max"] == pytest.approx(f1.max(), rel=1e-6)


def test_stats_are_keyed_by_name_so_reordering_inputs_is_safe(tmp_path) -> None:
    """The reason the file is not a positional list: swap `inputs` and each variable
    must still get ITS OWN mean, not the other one's."""
    root = _make_root(tmp_path)
    per_variable = _write_stats(root)

    dm = ZarrData(root=root, inputs=["feature_2", "feature_1"], targets=TARGETS)
    dm.setup("fit")

    assert dm.stats["x_mean"] == pytest.approx(
        [per_variable["feature_2"]["mean"], per_variable["feature_1"]["mean"]]
    )


def test_stats_for_an_unlisted_variable_is_a_clear_error(tmp_path) -> None:
    root = _make_root(tmp_path)
    (root / "stats.json").write_text(json.dumps({"feature_1": {"mean": 0.0, "std": 1.0}}))
    with pytest.raises(KeyError, match="utils/stats.py"):
        ZarrData(root=root, inputs=INPUTS, targets=TARGETS).setup("fit")


def test_stats_reach_the_model_buffers_and_then_the_checkpoint(tmp_path) -> None:
    root = _make_root(tmp_path)
    per_variable = _write_stats(root)

    module = MLP(in_dim=2, out_dim=1)
    assert module.x_norm.mean.tolist() == [0.0, 0.0], "precondition: identity"

    Trainer(max_steps=2, val_every=0, log_every=0, device="cpu").fit(
        module, ZarrData(root=root, inputs=INPUTS, targets=TARGETS, batch_size=8)
    )

    assert module.x_norm.mean.tolist() == pytest.approx(
        [per_variable["feature_1"]["mean"], per_variable["feature_2"]["mean"]], rel=1e-6
    )
    assert module.y_norm.mean.tolist() == pytest.approx(
        [per_variable["feature_3"]["mean"]], rel=1e-6
    )
    # Buffers, not attributes -- so they ride inside every checkpoint.
    assert "x_norm.mean" in module.state_dict()
    assert "y_norm.std" in module.state_dict()


def test_no_stats_file_is_a_warning_not_a_crash(tmp_path) -> None:
    """A missing stats.json must not block a first run; it leaves the identity."""
    dm = ZarrData(root=_make_root(tmp_path), inputs=INPUTS, targets=TARGETS, batch_size=8)
    dm.setup("fit")
    assert dm.stats == {}

    module = MLP(in_dim=2, out_dim=1)
    module.on_data_ready(dm)
    assert module.x_norm.std.tolist() == [1.0, 1.0]


# -- normalisation ------------------------------------------------------------------


def test_forward_takes_and_returns_raw_units(tmp_path) -> None:
    """The property inference depends on: feed physical units, get physical units.
    feature_3 is ~70, so an un-denormalised output would be nowhere near it."""
    root = _make_root(tmp_path, n_train=512, n_val=128)
    _write_stats(root)

    module = MLP(in_dim=2, out_dim=1, lr=1e-2)
    Trainer(max_steps=300, val_every=0, log_every=0, device="cpu").fit(
        module, ZarrData(root=root, inputs=INPUTS, targets=TARGETS, batch_size=32)
    )

    variables = _write_store(root / "val.zarr", 128, seed=1)
    x = torch.tensor([[variables["feature_1"][0], variables["feature_2"][0]]])
    with torch.no_grad():
        prediction = module(x)
    assert prediction.item() == pytest.approx(float(variables["feature_3"][0]), rel=0.05)


def test_normalisation_survives_a_checkpoint_round_trip(tmp_path) -> None:
    """Inference loads weights and nothing else -- no stats.json, no dataloader."""
    root = _make_root(tmp_path)
    _write_stats(root)

    trained = MLP(in_dim=2, out_dim=1)
    Trainer(max_steps=2, val_every=0, log_every=0, device="cpu").fit(
        trained, ZarrData(root=root, inputs=INPUTS, targets=TARGETS, batch_size=8)
    )

    fresh = MLP(in_dim=2, out_dim=1)  # never saw the data
    fresh.load_state_dict(trained.state_dict())

    x = torch.tensor([[101.0, -0.011]])
    with torch.no_grad():
        assert fresh(x).item() == pytest.approx(trained(x).item())
    assert fresh.x_norm.mean.tolist() == trained.x_norm.mean.tolist()


def test_denorm_inverts_norm() -> None:
    norm = MLP(in_dim=2, out_dim=1).x_norm
    norm.fit([100.0, -0.01], [5.0, 0.002])
    x = torch.tensor([[97.0, -0.013], [104.0, -0.004]])
    assert torch.allclose(norm.denorm(norm.norm(x)), x, atol=1e-4)


def test_a_constant_variable_does_not_divide_by_zero() -> None:
    """std is 0 for a constant column, which would normalise to inf."""
    norm = MLP(in_dim=1, out_dim=1).x_norm
    norm.fit([3.0], [0.0])
    assert torch.isfinite(norm.norm(torch.tensor([[3.0]]))).all()
