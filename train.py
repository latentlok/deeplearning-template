#!/usr/bin/env python3
"""Train.

    python train.py                                   # defaults
    python train.py experiment=e0                     # a saved recipe
    python train.py experiment=e0 model.hidden_dim=256 trainer.max_steps=5000
    python train.py -m experiment=e0 seed=1,2,3        # sweep
    python train.py --cfg job                          # print the composed config

Everything is reachable by dotted path. The run itself lives in engine/train.py and
you should not need to open it; @hydra.main has to be applied HERE because Hydra
locates configs/ relative to the __main__ module.
"""

import hydra
from omegaconf import DictConfig

from engine.train import run


@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig) -> float | None:
    return run(cfg)


if __name__ == "__main__":
    main()
