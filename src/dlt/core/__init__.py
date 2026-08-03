"""Infrastructure you rarely touch.

Extension points, all open:
  Trainer          -- subclass, override one method
  Logger           -- subclass for a new tracking backend
  WeightFormat     -- subclass for dtype-specific checkpointing
  Callback         -- observe; never own the update
"""

from dlt.core.base import (
    Batch,
    DataModule,
    Loaders,
    OptimSpec,
    SchedulerSpec,
    TaskModule,
    TrainState,
)
from dlt.core.trainer import Trainer

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
