"""Evaluation entrypoint: load a checkpoint, run one pass, log the metrics.

    python eval.py ckpt=outputs/pinn/2026-08-03_14-22-05_a3f9c2/ckpt/best_weights

Output lands in <that run>/eval/<timestamp>/, NOT in a global outputs/eval/ tree --
the metrics stay next to the config and the weights that produced them. The run
directory is derived from `ckpt` by the ckpt_run_dir resolver in configs/eval.yaml.

Deliberately thin. Long-horizon rollout testing is too model-specific to generalise,
so it stays yours -- write a script, and put its output in the run's artifacts/ dir
via engine.tracking.artifacts_dir so it cannot drift from the weights that made it.

Note what is NOT called here: module.on_data_ready(). An evaluation takes its
normalisation statistics from the checkpoint. Recomputing them from whatever data is
mounted today is the exact train/inference desync that hook exists to prevent.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from engine.checkpoint import load_checkpoint
from engine.tracking import ConsoleLogger, JSONLLogger, MultiLogger, TensorBoardLogger
from engine.utils import register_resolvers, resolve_precision, seed_everything, set_default_dtype

register_resolvers()

log = logging.getLogger(__name__)


def run(cfg: DictConfig) -> dict[str, float]:
    """The body of an evaluation. @hydra.main lives on ../eval.py -- see engine/train.py."""
    if not cfg.get("ckpt"):
        raise ValueError("set ckpt=<path to a checkpoint directory>, e.g. .../ckpt/best")

    run_dir = Path(HydraConfig.get().runtime.output_dir)
    param_dtype, amp_dtype = resolve_precision(cfg.dtype, cfg.amp)
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    set_default_dtype(param_dtype)

    logger = MultiLogger(
        [
            TensorBoardLogger(run_dir / "tb") if cfg.tracking.tensorboard else None,
            JSONLLogger(run_dir / "metrics.jsonl") if cfg.tracking.jsonl else None,
            ConsoleLogger(every=1) if cfg.tracking.console else None,
        ]
    )

    try:
        datamodule = hydra.utils.instantiate(cfg.data)
        module = hydra.utils.instantiate(cfg.model)
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            callbacks=[],
            param_dtype=param_dtype,
            amp_dtype=amp_dtype,
        )

        # weights_only: an eval does not want a stale optimizer or step counter.
        load_checkpoint(cfg.ckpt, module, weights_only=True)

        datamodule.setup("validate")
        module.to(device=trainer.device, dtype=param_dtype)
        trainer.module, trainer.datamodule = module, datamodule

        prefix = "rollout" if cfg.step == "rollout_step" else "val"
        trainer._run_eval(cfg.step, prefix, 0)

        log.info("metrics: %s", trainer.state.metrics)
        return trainer.state.metrics
    finally:
        logger.close()
