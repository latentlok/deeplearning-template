"""The orchestration. You should be able to work here for months without opening it.

    base.py        the two contracts: TaskModule, DataModule
    trainer.py     the loop
    callbacks.py   checkpointing, early stopping, gradient stats -- observers only
    checkpoint.py  save / load / resume
    tracking.py    TensorBoard, JSONL, console, run metadata
    utils.py       seeding, precision, device movement, schedules, run naming
    train.py       the train entrypoint (../train.py is a two-line shim)
    eval.py        the eval entrypoint

Extension points, all open:
  Trainer          -- subclass, override one method
  Logger           -- subclass for a new tracking backend
  WeightFormat     -- subclass for dtype-specific checkpointing
  Callback         -- observe; never own the update
"""

from engine.base import (
    Batch,
    DataModule,
    Loaders,
    OptimSpec,
    SchedulerSpec,
    TaskModule,
    TrainState,
)
from engine.trainer import Trainer

__all__ = [
    "Batch",
    "DataModule",
    "Loaders",
    "OptimSpec",
    "SchedulerSpec",
    "TaskModule",
    "TrainState",
    "Trainer",
]
