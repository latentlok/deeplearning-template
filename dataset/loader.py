"""The dataloader you rewrite. Reads data that lives OUTSIDE the repo.

Expected layout, under ${paths.data_root} (env var DL_DATA):

    <data_root>/
        train/  x.npy   y.npy        # or x.zarr / y.zarr
        val/    x.npy   y.npy
        stats.json                   # written by utils/stats.py -- optional

Arrays are opened LAZILY: .npy through a numpy memmap, .zarr through zarr's own lazy
indexing. Nothing is read until a sample is requested, so a 40 GB training set costs
no RAM at startup and `num_workers` scales the reads.

Three things here are deliberate, and each is a bug you would otherwise hit:

  1. setup() NEVER scans the data to compute statistics. It loads them from
     stats.json. Scanning would add minutes to every run, and -- far worse -- the val
     split would normalise against different numbers than the train split. Run
     `python utils/stats.py` once, commit the json alongside the data.
  2. Samples are returned in their stored dtype. The Trainer casts floating-point
     tensors to the run's dtype when it moves the batch to the device, so an fp64 PINN
     run and an fp32 run share this file unchanged.
  3. Batches are dicts. The Trainer never inspects a batch, so the structure is yours
     -- add a "mask" or "meta" key and it flows through to your training_step
     untouched.

To adapt this to your data, the interesting lines are `_open` (how a shard is read),
`__getitem__` (what one sample is) and `_collate` (what a batch is). The rest is
plumbing you can leave alone.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from engine.base import DataModule

log = logging.getLogger(__name__)


def _open(path: Path) -> Any:
    """Open one array lazily. Suffix dispatch -- add your format here.

    Returns anything that supports len() and integer indexing.
    """
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    if path.suffix == ".zarr":
        try:
            import zarr
        except ImportError as e:  # not a dependency of the template
            raise ImportError(f"reading {path.name} needs zarr: uv add zarr") from e
        return zarr.open(str(path), mode="r")
    if path.suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError(f"unsupported array format: {path.name}")


def _find(directory: Path, stem: str) -> Path:
    """<dir>/<stem>.{npy,zarr,pt} -- whichever exists."""
    for suffix in (".npy", ".zarr", ".pt", ".pth"):
        if (p := directory / f"{stem}{suffix}").exists():
            return p
    raise FileNotFoundError(
        f"no {stem}.{{npy,zarr,pt}} in {directory}. Expected <data_root>/<split>/"
        f"{stem}.npy; set data.root / data.inputs / data.targets if your layout differs."
    )


class ArrayPairs(Dataset):
    """One (inputs, targets) pair per index. Both arrays must agree on length."""

    def __init__(self, x: Any, y: Any) -> None:
        if len(x) != len(y):
            raise ValueError(f"inputs and targets disagree on length: {len(x)} vs {len(y)}")
        self.x, self.y = x, y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        # np.asarray materialises just this sample out of the memmap / zarr chunk.
        # .copy() because torch.as_tensor on a memmap slice keeps the mapping alive,
        # which leaks file handles across worker processes.
        return {
            "x": torch.as_tensor(np.asarray(self.x[i]).copy()),
            "y": torch.as_tensor(np.asarray(self.y[i]).copy()),
        }


class FolderData(DataModule):
    """<root>/train and <root>/val, each holding one inputs array and one targets array."""

    def __init__(
        self,
        root: str | Path = "data",
        train_split: str = "train",
        val_split: str = "val",
        inputs: str = "x",
        targets: str = "y",
        stats: str | Path | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        # __init__ touches NO files: every config in configs/data/ is instantiated by
        # the test suite, on machines where the dataset is not mounted.
        self.root = Path(root).expanduser()
        self.train_split, self.val_split = train_split, val_split
        self.inputs, self.targets = inputs, targets
        self.stats_path = Path(stats).expanduser() if stats else None
        self.batch_size, self.num_workers = batch_size, num_workers
        self.pin_memory, self.drop_last = pin_memory, drop_last
        self.stats: dict[str, Any] = {}
        self.train_ds: ArrayPairs | None = None
        self.val_ds: ArrayPairs | None = None

    # -- setup ---------------------------------------------------------------------

    def _split(self, name: str) -> ArrayPairs:
        d = self.root / name
        if not d.is_dir():
            raise FileNotFoundError(
                f"split directory not found: {d}. The data lives outside the repo -- "
                f"point DL_DATA (or paths.data_root) at the right place."
            )
        return ArrayPairs(_open(_find(d, self.inputs)), _open(_find(d, self.targets)))

    def _load_stats(self) -> None:
        """Statistics are READ, never computed. See the module docstring."""
        path = self.stats_path or self.root / "stats.json"
        if not path.exists():
            log.warning(
                "no %s -- the model gets no normalisation bounds. Run "
                "`python utils/stats.py --root %s` to compute them once.",
                path,
                self.root,
            )
            return
        self.stats = json.loads(path.read_text())
        log.info("loaded stats from %s (%s)", path, ", ".join(sorted(self.stats)))

    def setup(self, stage: str) -> None:
        self._load_stats()
        if stage == "fit":
            self.train_ds = self._split(self.train_split)
        self.val_ds = self._split(self.val_split)

    # -- loaders -------------------------------------------------------------------

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=_collate,
            # Workers are re-forked every epoch by default, which re-opens every
            # memmap. Keeping them alive matters as soon as num_workers > 0.
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "call setup('fit') first"
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        # Never shuffled: a val metric must be comparable across evaluations.
        return self._loader(self.val_ds, shuffle=False) if self.val_ds is not None else None


def _collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """List of samples -> one batch. Augmentation and masking belong HERE (or in
    training_step), never in a callback."""
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
