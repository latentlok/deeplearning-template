"""Evaluation entrypoint: load a checkpoint, run one pass, log the metrics.

Deliberately thin. Long-horizon rollout testing is too model-specific to generalise,
so it stays yours -- write a script, and put its output in the run's artifacts/ dir
via dlt.core.tracking.artifacts_dir so it cannot drift from the weights that made it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from dlt.core.checkpoint import load_checkpoint
from dlt.core.tracking import ConsoleLogger, JSONLLogger, MultiLogger, TensorBoardLogger
from dlt.core.utils import register_resolvers, resolve_precision, seed_everything, set_default_dtype

register_resolvers()

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval")
def main(cfg: DictConfig) -> dict[str, float]:
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


if __name__ == "__main__":
    main()
