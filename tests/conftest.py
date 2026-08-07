from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from engine.utils import register_resolvers

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"

register_resolvers()


def config_names(group: str) -> list[str]:
    """Every yaml in a config group. New files are picked up automatically -- this is
    what makes adding a model cost zero test code."""
    d = CONFIGS / group
    return sorted(p.stem for p in d.glob("*.yaml")) if d.is_dir() else []


def load(overrides: list[str] | None = None):
    with initialize_config_dir(config_dir=str(CONFIGS), version_base="1.3"):
        return compose(
            config_name="train",
            overrides=(overrides or []),
            return_hydra_config=True,
        )


@pytest.fixture(scope="session")
def cfg():
    return load(["experiment=e0", "trainer.device=cpu"])
