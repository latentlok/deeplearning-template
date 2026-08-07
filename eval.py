#!/usr/bin/env python3
"""Evaluate a checkpoint.

    python eval.py ckpt=outputs/e0/2026-08-03_14-22-05_a3f9c2/ckpt/best_weights
    python eval.py ckpt=.../ckpt/best step=rollout_step

Metrics are written into that run's own directory, under eval/<timestamp>/, so they
cannot drift from the config and weights that produced them.
"""

import hydra
from omegaconf import DictConfig

from engine.eval import run


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> dict[str, float]:
    return run(cfg)


if __name__ == "__main__":
    main()
