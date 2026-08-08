"""The dataloader you rewrite. Reads a zarr group that lives OUTSIDE the repo.

Expected layout, under ${paths.data_root} (env var DL_DATA):

    <data_root>/
        train.zarr/     time (T,)  feature_1 (T,)  feature_2 (T,)  feature_3 (T,)
        val.zarr/       same variables
        stats.json      # written by utils/stats.py -- optional

A zarr group is a directory of named arrays, so which variables are inputs and which
are targets is a config decision, not a storage decision:

    inputs:  [feature_1, feature_2]
    targets: [feature_3]

Variables you do not name are never read -- `time` is there for orientation and costs
nothing to leave out.

Sample i is row i: the input variables at index i concatenated into one vector, and
the target variables at the same index. To sample differently -- a window of rows,
every k-th row -- rewrite `ZarrRows` and nothing else.

Nothing is read until a sample is requested: zarr decompresses only the chunk holding
row i, so a store far larger than RAM costs nothing at startup.

Three things here are deliberate:

  1. setup() NEVER scans the data to compute statistics. It reads stats.json. Scanning
     would add minutes to every run and -- far worse -- the val split would normalise
     against different numbers than train. Run `python utils/stats.py` once.
  2. stats.json is keyed BY VARIABLE NAME. Reordering or subsetting `inputs` in config
     therefore cannot silently pair a variable with another variable's mean.
  3. Samples are returned in their stored dtype. The Trainer casts floating-point
     tensors to the run's dtype when it moves the batch to the device.

Two zarr facts that are not obvious from its docs:

  * A zarr 3 Array has NO __len__. Use `.shape[0]`; `len(array)` raises TypeError.
  * Indexing a (T,) array with an int gives a 0-d array, not a scalar, so np.atleast_1d
    is what makes a (T,) variable and a (T, D) variable concatenate the same way.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from engine.base import DataModule

log = logging.getLogger(__name__)


def open_group(path: Path) -> Any:
    """Open a zarr group read-only, with errors that say what to fix."""
    import zarr

    try:
        return zarr.open_group(str(path), mode="r")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"no zarr store at {path}. The data lives outside the repo -- point "
            f"DL_DATA (or paths.data_root) at the directory holding train.zarr."
        ) from e
    except zarr.errors.ContainsArrayError as e:
        raise ValueError(
            f"{path} is a zarr array, not a group. This loader expects a group of "
            f"named variables, one array per variable."
        ) from e


def pick(group: Any, names: Sequence[str]) -> list[Any]:
    """Look up each named variable, listing what is in the store on a miss."""
    if not names:
        raise ValueError("no variables named -- set data.inputs / data.targets")
    if missing := [n for n in names if n not in group]:
        available = ", ".join(sorted(group.array_keys())) or "<none>"
        raise KeyError(f"variable(s) {missing} not in store. Available: {available}")
    return [group[n] for n in names]


def row(arrays: list[Any], i: int) -> torch.Tensor:
    """Row i of every variable, concatenated into one flat vector."""
    return torch.as_tensor(np.concatenate([np.atleast_1d(np.asarray(a[i])) for a in arrays]))


def _common_length(arrays: dict[str, Any]) -> int:
    """Every variable must span the same number of rows. A mismatch means the store was
    written by two different runs, which is otherwise silent until it indexes past the
    end of the shorter one partway through an epoch."""
    lengths = {name: a.shape[0] for name, a in arrays.items()}  # zarr Arrays have no len()
    if len(set(lengths.values())) != 1:
        raise ValueError(f"variables disagree on length: {lengths}")
    return next(iter(lengths.values()))


class ZarrRows(Dataset):
    """One row per index. THIS is the class to rewrite for a different sampling scheme:
    a windowed dataset changes __len__ and __getitem__ and touches nothing else."""

    def __init__(self, group: Any, inputs: Sequence[str], targets: Sequence[str]) -> None:
        self.x = pick(group, inputs)
        self.y = pick(group, targets)
        self.n = _common_length(dict(zip(inputs, self.x, strict=True)))
        if (m := _common_length(dict(zip(targets, self.y, strict=True)))) != self.n:
            raise ValueError(f"inputs and targets disagree on length: {self.n} vs {m}")

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"x": row(self.x, i), "y": row(self.y, i)}


class ZarrData(DataModule):
    """<root>/train.zarr and <root>/val.zarr, each a group of named variables."""

    def __init__(
        self,
        root: str | Path = "data",
        train_store: str = "train.zarr",
        val_store: str = "val.zarr",
        inputs: Sequence[str] = ("feature_1", "feature_2"),
        targets: Sequence[str] = ("feature_3",),
        stats: str | Path | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        # __init__ touches NO files: tests/test_configs.py instantiates every config in
        # configs/data/ on machines where the dataset is not mounted.
        self.root = Path(root).expanduser()
        self.train_store, self.val_store = train_store, val_store
        self.inputs, self.targets = list(inputs), list(targets)
        self.stats_path = Path(stats).expanduser() if stats else None
        self.batch_size, self.num_workers = batch_size, num_workers
        self.pin_memory, self.drop_last = pin_memory, drop_last
        self.stats: dict[str, Any] = {}
        self.train_ds: ZarrRows | None = None
        self.val_ds: ZarrRows | None = None

    def _split(self, store: str) -> ZarrRows:
        return ZarrRows(open_group(self.root / store), self.inputs, self.targets)

    def _load_stats(self) -> None:
        """Read stats.json and assemble it into the vectors the model wants.

        On disk the file is keyed by variable name. Here it becomes x_mean / x_std /
        y_mean / y_std, concatenated in the order `inputs` and `targets` name them --
        the same order __getitem__ concatenates the row, which is what makes the two
        line up. The model gets vectors and never has to know about names.
        """
        path = self.stats_path or self.root / "stats.json"
        if not path.exists():
            log.warning(
                "no %s -- the model gets no normalisation and trains on raw units. "
                "Run `python utils/stats.py --root %s` to compute them once.",
                path,
                self.root,
            )
            return
        per_variable = json.loads(path.read_text())
        self.stats = {
            f"{side}_{key}": _assemble(per_variable, names, key, path)
            for side, names in (("x", self.inputs), ("y", self.targets))
            for key in ("mean", "std")
        }
        log.info("loaded stats from %s for %s", path, ", ".join(self.inputs + self.targets))

    def setup(self, stage: str) -> None:
        self._load_stats()
        if stage == "fit":
            self.train_ds = self._split(self.train_store)
        self.val_ds = self._split(self.val_store)

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            collate_fn=_collate,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        assert self.train_ds is not None, "call setup('fit') first"
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader | None:
        # Never shuffled: a val metric must be comparable across evaluations.
        return self._loader(self.val_ds, shuffle=False) if self.val_ds is not None else None


def _assemble(per_variable: dict, names: Sequence[str], key: str, path: Path) -> list[float]:
    """Concatenate one statistic across the named variables, in config order."""
    missing = [n for n in names if n not in per_variable]
    if missing:
        raise KeyError(
            f"{path} has no statistics for {missing}. It is keyed by variable name; "
            f"rerun `python utils/stats.py` after changing data.inputs / data.targets."
        )
    return [float(v) for n in names for v in np.atleast_1d(per_variable[n][key])]


def _collate(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """List of samples -> one batch. Augmentation and masking belong HERE (or in
    training_step), never in a callback."""
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}
