"""Training entrypoint.

Returns the monitored metric, which is the entire coupling surface an HPO sweeper
needs. No sweeper is shipped; adding one later touches nothing in src/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from engine.tracking import (
    ConsoleLogger,
    JSONLLogger,
    MultiLogger,
    RunMeta,
    TensorBoardLogger,
    flatten_config,
)
from engine.utils import (
    config_hash,
    is_rank_zero,
    register_resolvers,
    resolve_precision,
    seed_everything,
    set_default_dtype,
)

# Must run before @hydra.main composes -- hydra.run.dir interpolates ${run_hash:...}.
register_resolvers()

log = logging.getLogger(__name__)


def build_logger(cfg: DictConfig, run_dir: Path) -> MultiLogger:
    """Three channels, independently toggleable. TensorBoard is not always viewable;
    train.log and metrics.jsonl are what keep a headless run legible."""
    # Gate CONSTRUCTION, not just the calls: MultiLogger drops writes off rank zero,
    # but SummaryWriter creates its event file in __init__, so every rank left a
    # second, empty event file in tb/ and `tensorboard --logdir` showed phantom runs.
    if not is_rank_zero():
        return MultiLogger([])
    loggers = []
    if cfg.tracking.tensorboard:
        loggers.append(TensorBoardLogger(run_dir / "tb"))
    if cfg.tracking.jsonl:
        loggers.append(JSONLLogger(run_dir / "metrics.jsonl"))
    if cfg.tracking.console:
        loggers.append(ConsoleLogger(every=cfg.tracking.console_every))
    return MultiLogger(loggers)


def run(cfg: DictConfig) -> float | None:
    """The body of a training run. @hydra.main lives on ../train.py, not here.

    Hydra derives its config search path from the module that owns the decorated
    function: only when that module is `__main__` does it use the file's directory.
    Decorate an imported function and it looks for a `configs` PACKAGE instead and
    fails with "Primary config module 'configs' not found".
    """
    run_dir = Path(HydraConfig.get().runtime.output_dir)

    # Every rank shares one run dir, so INFO from N ranks means N copies of every line
    # in train.log. Warnings and tracebacks still come through from every rank -- the
    # one that dies is rarely rank zero.
    if not is_rank_zero():
        logging.getLogger().setLevel(logging.WARNING)

    # dtype and amp are orthogonal axes; validate the pair before anything is built so
    # a conflict is a config error, not a failure deep in a backward pass.
    param_dtype, amp_dtype = resolve_precision(cfg.dtype, cfg.amp)

    # Both UNCONDITIONAL: Hydra's basic launcher runs multirun jobs sequentially in one
    # process, so anything set conditionally leaks from job to job. set_default_dtype
    # must also precede model construction, or the model is fp32 while the data is fp64.
    seed_everything(cfg.seed, deterministic=cfg.deterministic)
    set_default_dtype(param_dtype)

    log.info("run dir: %s", run_dir)
    meta = RunMeta(run_dir, cfg.exp_name, config_hash(cfg))
    logger = build_logger(cfg, run_dir)
    logger.log_text("config", f"```yaml\n{OmegaConf.to_yaml(cfg, resolve=True)}\n```")

    try:
        datamodule = hydra.utils.instantiate(cfg.data)
        module = hydra.utils.instantiate(cfg.model)
        callbacks = [hydra.utils.instantiate(c) for c in (cfg.callbacks or {}).values()]
        trainer = hydra.utils.instantiate(
            cfg.trainer,
            logger=logger,
            callbacks=callbacks,
            param_dtype=param_dtype,
            amp_dtype=amp_dtype,
        )

        # Resume is handled inside fit(), which builds the optimizers -- loading here
        # would populate optimizers that fit() then replaces, silently continuing with
        # a cold optimizer.
        result = trainer.fit(module, datamodule, resume=cfg.get("resume"))

        # add_hparams at the end is what makes the HPARAMS tab work and runs comparable.
        logger.log_hparams(flatten_config(cfg), trainer.state.metrics)
        meta.finish("finished", trainer.state.metrics)
        log.info("%s = %s", cfg.trainer.monitor, result)
        return result

    except BaseException as e:
        # Crashed runs must be visibly crashed, not silently absent. The traceback goes
        # to train.log, not just the terminal you already closed.
        log.exception("run failed: %s", type(e).__name__)
        meta.finish("failed")
        raise
    finally:
        logger.close()
