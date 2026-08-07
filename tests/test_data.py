"""The two files a fork actually rewrites: dataset/loader.py and utils/stats.py.

The dataset here is four .npy files in tmp_path, which is what FolderData expects to
find under DL_DATA. The .zarr path is NOT exercised -- zarr is not a dependency.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from dataset.loader import FolderData
from engine.trainer import Trainer
from models.mlp import MLP
from utils.stats import compute_stats


def _make_dataset(root, n_train: int = 64, n_val: int = 16, dim: int = 4):
    rng = np.random.default_rng(0)
    w = rng.normal(size=(dim, 1))
    for split, n in (("train", n_train), ("val", n_val)):
        d = root / split
        d.mkdir(parents=True)
        x = rng.normal(size=(n, dim)).astype("float32")
        np.save(d / "x.npy", x)
        np.save(d / "y.npy", (x @ w).astype("float32"))
    return root


def test_folder_data_trains_end_to_end(tmp_path) -> None:
    dm = FolderData(root=_make_dataset(tmp_path), batch_size=8)
    trainer = Trainer(max_steps=5, val_every=5, log_every=0, device="cpu")
    result = trainer.fit(MLP(in_dim=4, out_dim=1), dm)
    assert result is not None and torch.isfinite(torch.tensor(result))
    assert trainer.state.global_step == 5


def test_missing_split_points_at_the_data_root(tmp_path) -> None:
    """The most common first failure is a mis-set DL_DATA, so the error says so."""
    with pytest.raises(FileNotFoundError, match="DL_DATA"):
        FolderData(root=tmp_path / "nowhere").setup("fit")


def test_stats_match_numpy(tmp_path) -> None:
    """Streaming accumulation must agree with the obvious one-shot computation."""
    root = _make_dataset(tmp_path, n_train=100, dim=3)
    x = np.load(root / "train" / "x.npy")
    stats = compute_stats(x, axis=1, chunk=7)  # chunk deliberately not a divisor of 100

    assert stats["count"] == 100
    assert stats["lower"] == pytest.approx(x.min(axis=0).tolist(), rel=1e-6)
    assert stats["upper"] == pytest.approx(x.max(axis=0).tolist(), rel=1e-6)
    assert stats["mean"] == pytest.approx(x.mean(axis=0).tolist(), rel=1e-5)
    assert stats["std"] == pytest.approx(x.std(axis=0).tolist(), rel=1e-5)


def test_stats_reach_the_model_and_then_the_checkpoint(tmp_path) -> None:
    """The chain this whole layout exists to protect:

        train split -> stats.json -> FolderData.stats -> model buffers -> state_dict

    A break anywhere in it is silent: the model still trains, and inference normalises
    against different numbers than training did.
    """
    root = _make_dataset(tmp_path)
    x = np.load(root / "train" / "x.npy")
    stats = compute_stats(x)
    (root / "stats.json").write_text(json.dumps(stats))

    module = MLP(in_dim=4, out_dim=1)
    assert module.lower.tolist() == [0.0] * 4, "precondition: identity transform"

    Trainer(max_steps=2, val_every=0, log_every=0, device="cpu").fit(
        module, FolderData(root=root, batch_size=8)
    )

    assert module.lower.tolist() == pytest.approx(stats["lower"], rel=1e-6)
    assert module.upper.tolist() == pytest.approx(stats["upper"], rel=1e-6)
    # Buffers, not attributes -- so they ride inside every checkpoint.
    assert "lower" in module.state_dict()


def test_no_stats_file_is_a_warning_not_a_crash(tmp_path) -> None:
    """A missing stats.json must not block a first run; it leaves the identity."""
    dm = FolderData(root=_make_dataset(tmp_path), batch_size=8)
    dm.setup("fit")
    assert dm.stats == {}

    module = MLP(in_dim=4, out_dim=1)
    module.on_data_ready(dm)
    assert module.upper.tolist() == [1.0] * 4
