"""Every config must compose AND instantiate.

The highest-value test in a Hydra project: it catches a broken defaults list or a
renamed argument before you burn GPU hours, and it needs no new code when you add a
config.
"""

from __future__ import annotations

import hydra
import pytest

from tests.conftest import CONFIGS, config_names, load


@pytest.mark.parametrize("name", config_names("experiment"))
def test_experiment_composes(name: str) -> None:
    cfg = load([f"experiment={name}"])
    assert cfg.exp_name
    assert cfg.model._target_
    assert cfg.data._target_


@pytest.mark.parametrize("name", config_names("model"))
def test_model_instantiates(name: str) -> None:
    cfg = load([f"model={name}"])
    hydra.utils.instantiate(cfg.model)


@pytest.mark.parametrize("name", config_names("data"))
def test_data_instantiates(name: str) -> None:
    cfg = load([f"data={name}"])
    hydra.utils.instantiate(cfg.data)


@pytest.mark.parametrize("name", config_names("optim"))
def test_optim_builds_an_optimizer(name: str) -> None:
    """optim configs are _partial_: they instantiate to a factory, and only become an
    optimizer once a model hands them its parameters."""
    import torch

    factory = hydra.utils.instantiate(load([f"optim={name}"]).optim)
    opt = factory([torch.nn.Parameter(torch.zeros(2))])
    assert isinstance(opt, torch.optim.Optimizer)


@pytest.mark.parametrize("name", config_names("sched"))
def test_sched_builds_against_an_optimizer(name: str) -> None:
    import torch

    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(2))], lr=0.1)
    sched = hydra.utils.instantiate(load([f"sched={name}"]).sched)(opt)
    assert hasattr(sched, "step") and hasattr(sched, "state_dict")


def test_the_model_takes_its_optimizer_from_the_optim_group() -> None:
    """The whole point of the group: `optim=sgd` must reach the built optimizer without
    editing any model file."""
    import torch

    cfg = load(["model=mlp", "optim=sgd", "optim.lr=0.123"])
    spec = hydra.utils.instantiate(cfg.model).configure_optimizers()
    assert isinstance(spec.optimizers[0], torch.optim.SGD)
    assert spec.optimizers[0].param_groups[0]["lr"] == pytest.approx(0.123)


def test_no_scheduler_unless_one_is_asked_for() -> None:
    """`sched: null` is the default, and oc.select must resolve it to None rather than
    raising on a missing interpolation."""
    spec = hydra.utils.instantiate(load(["model=mlp"]).model).configure_optimizers()
    assert spec.schedulers == []

    spec = hydra.utils.instantiate(load(["model=mlp", "sched=cosine"]).model).configure_optimizers()
    assert len(spec.schedulers) == 1


@pytest.mark.parametrize("name", config_names("debug"))
def test_debug_composes(name: str) -> None:
    load([f"debug={name}"])


def test_run_dir_template_is_grouped_and_hashed() -> None:
    """outputs/<exp>/<timestamp>_<hash6>.

    The unresolved template is asserted, not the resolved path: HydraConfig and
    hydra.job.override_dirname only exist at job launch, so resolving here would fail
    for reasons unrelated to the naming scheme.
    """
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(CONFIGS / "train.yaml"), resolve=False)
    template = raw["hydra"]["run"]["dir"]
    assert "${exp_name}" in template, "runs must be grouped by experiment"
    assert "${now:" in template, "runs must be time-sortable"
    assert "${run_hash:" in template, "identical configs must be identifiable"
    assert raw["hydra"]["job"]["chdir"] is False, "chdir must stay false"


def test_paths_never_chain_to_a_hydra_resolver() -> None:
    """${hydra:...} resolves only inside a launched job. hydra.sweep.dir is built
    before HydraConfig exists (so --multirun dies), and hydra.compose() has no
    HydraConfig at all (so every config test dies). Both are silent for single runs,
    which is what makes this worth asserting."""
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(CONFIGS / "paths" / "default.yaml"), resolve=False)
    for key, value in raw.items():
        assert "${hydra:" not in str(value), f"paths.{key} must stay hydra-free: {value}"


def test_eval_writes_into_the_run_that_produced_the_checkpoint() -> None:
    """A global outputs/eval/ tree is orphaned the moment an experiment has two runs:
    the metrics are real but nothing on disk says which weights made them."""
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(OmegaConf.load(CONFIGS / "eval.yaml"), resolve=False)
    template = raw["hydra"]["run"]["dir"]
    assert "${ckpt_run_dir:${ckpt}}" in template, "eval must resolve its dir from the ckpt"
    assert "${paths.output_root}" not in template, "eval must not open a parallel tree"


def test_ckpt_run_dir_walks_up_to_the_run() -> None:
    from engine.utils import ckpt_run_dir

    run = "outputs/pinn/2026-08-03_14-22-05_a3f9c2"
    assert ckpt_run_dir(f"{run}/ckpt/best") == run
    assert ckpt_run_dir(f"{run}/ckpt/step_00001000") == run
    # No ckpt/ ancestor: still beside the weights, never in a global bucket.
    assert ckpt_run_dir("/tmp/downloaded/model") == "/tmp/downloaded"
    with pytest.raises(ValueError, match="ckpt="):
        ckpt_run_dir(None)


def test_run_hash_is_stable_and_override_sensitive() -> None:
    from engine.utils import run_hash

    assert run_hash("experiment=e0") == run_hash("experiment=e0")
    # Order must not matter -- the same config should not produce two directories.
    assert run_hash("a=1,b=2") == run_hash("b=2,a=1")
    assert run_hash("experiment=e0") != run_hash("experiment=e0,model.hidden_dim=999")
    assert len(run_hash("x=1")) == 6


def test_amp_requires_fp32() -> None:
    """The dtype/amp pair is validated at startup, not deep in a backward pass."""
    from engine.utils import resolve_precision

    resolve_precision("float32", "bf16")
    resolve_precision("float64", "none")
    with pytest.raises(ValueError, match="requires dtype='float32'"):
        resolve_precision("float64", "bf16")
