"""A gradient loss with adaptive term weighting.

Problem: du/dx = -u with u(0) = 1, whose exact solution is e^(-x). Chosen because the
physics is ~15 lines, it needs no dataset (collocation points are torch.rand), and it
has an ANALYTIC solution -- so the test can assert the model actually converges. An
unverifiable physics example is worse than none.

It also carries the imbalance that motivates adaptive weighting: the residual is
averaged over hundreds of collocation points, the boundary over one. Untuned, the
residual dominates and the network happily learns u ~ 0, which satisfies du/dx = -u
perfectly and ignores the boundary. You can watch that happen in TensorBoard.

The gradient-loss mechanism works because requires_grad is set BEFORE the forward
pass, in the step method -- the module is the thing that knows it needs derivatives.
A (pred, target) loss signature would make this structurally impossible.

Its data half is CollocationData in dataset/examples.py -- collocation points are
torch.rand, so there is nothing to download.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from engine.base import OptimSpec, TaskModule, TrainState

# ---------------------------------------------------------------------------
# The weighting seam. An nn.Module, not a function -- that is load-bearing:
#   * learnable weights are just nn.Parameter on it
#   * update() can own its own optimizer and run an inner loop
#   * as a submodule it is CHECKPOINTED and RESUMED automatically. Resume with
#     silently reset loss weights and your continued run is not the run you think.
# It lives here rather than in engine/ because weighting is a per-project concern.
# ---------------------------------------------------------------------------


class Weighting(nn.Module, ABC):
    def __init__(self, names: list[str], init: float = 1.0) -> None:
        super().__init__()
        self.names = list(names)
        self.register_buffer("weights", torch.full((len(self.names),), float(init)))

    def w(self, name: str) -> Tensor:
        return self.weights[self.names.index(name)]

    @abstractmethod
    def update(self, terms: dict[str, Tensor], params: list[Tensor], state: TrainState) -> None: ...

    def combine(self, terms: dict[str, Tensor]) -> Tensor:
        return sum(self.w(k) * v for k, v in terms.items())  # type: ignore[return-value]

    def as_metrics(self) -> dict[str, Tensor]:
        return {f"weight_{n}": self.weights[i] for i, n in enumerate(self.names)}


class StaticWeights(Weighting):
    def __init__(self, names: list[str], values: dict[str, float] | None = None) -> None:
        super().__init__(names)
        for k, v in (values or {}).items():
            self.weights[self.names.index(k)] = float(v)

    def update(self, terms: dict[str, Tensor], params: list[Tensor], state: TrainState) -> None:
        """No-op. Weights are whatever you configured."""


class GradNormWeights(Weighting):
    """Wang-style learning-rate annealing: balance terms by gradient magnitude.

        lambda_hat_i = ||grad L_ref|| / ||grad L_i||
        lambda_i     = alpha * lambda_i + (1 - alpha) * lambda_hat_i

    Costs one autograd.grad per term, hence `every`. Weights are detached constants --
    nothing backpropagates through them.

    This needs NO change to the core contract: training_step still returns a scalar
    loss, and retain_graph=True here keeps the graph alive for the Trainer's backward.
    """

    def __init__(
        self,
        names: list[str],
        reference: str,
        every: int = 100,
        alpha: float = 0.9,
        eps: float = 1e-8,
        max_weight: float = 1e4,
    ) -> None:
        super().__init__(names)
        self.reference, self.every = reference, every
        self.alpha, self.eps, self.max_weight = alpha, eps, max_weight

    @torch.no_grad()
    def _blend(self, name: str, new: float) -> None:
        i = self.names.index(name)
        self.weights[i] = self.alpha * self.weights[i] + (1 - self.alpha) * min(
            new, self.max_weight
        )

    def update(self, terms: dict[str, Tensor], params: list[Tensor], state: TrainState) -> None:
        if not self.every or state.global_step % self.every:
            return
        norms: dict[str, float] = {}
        for name, term in terms.items():
            grads = torch.autograd.grad(
                term, params, retain_graph=True, allow_unused=True, create_graph=False
            )
            total = sum(float(g.detach().pow(2).sum()) for g in grads if g is not None)
            norms[name] = total**0.5
        ref = norms.get(self.reference, 0.0)
        if ref <= 0.0:
            return
        for name, n in norms.items():
            self._blend(name, ref / (n + self.eps))


# ---------------------------------------------------------------------------


class PINN(TaskModule):
    # Evaluation must NOT be wrapped in no_grad -- the residual needs autograd.
    eval_requires_grad = True

    def __init__(
        self,
        hidden_dim: int = 32,
        depth: int = 3,
        optim: Callable[..., Optimizer] | None = None,
        sched: Callable[..., Any] | None = None,
        lr: float = 1e-3,
        weighting: Weighting | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden_dim), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers += [nn.Linear(hidden_dim, 1)]
        self.net = nn.Sequential(*layers)
        # Factories from configs/optim/ and configs/sched/; `lr` is the fallback for
        # direct construction only.
        self.optim, self.sched, self.lr = optim, sched, lr
        # A submodule, so its weights are checkpointed and resumed with the model.
        self.weighting = weighting or StaticWeights(["residual", "boundary"])

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    def _terms(self, x: Tensor) -> dict[str, Tensor]:
        x = x.requires_grad_(True)  # BEFORE the forward pass
        u = self(x)
        (du_dx,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)

        residual = ((du_dx + u) ** 2).mean()  # over N collocation points
        zero = torch.zeros(1, 1, dtype=x.dtype, device=x.device)
        boundary = ((self(zero) - 1.0) ** 2).mean()  # over 1 point
        return {"residual": residual, "boundary": boundary}

    def training_step(self, batch: dict, state: TrainState) -> dict[str, Tensor]:
        terms = self._terms(batch["x"])
        self.weighting.update(terms, [p for p in self.net.parameters() if p.requires_grad], state)
        loss = self.weighting.combine(terms)
        return {
            "loss": loss,
            **{f"loss_{k}": v.detach() for k, v in terms.items()},
            **self.weighting.as_metrics(),
        }

    def validation_step(self, batch: dict, state: TrainState) -> dict[str, Tensor]:
        x = batch["x"]
        terms = self._terms(x)
        with torch.no_grad():
            # The point of an analytic solution: a metric that cannot be gamed.
            err = (self(x) - torch.exp(-x)).abs().mean()
        return {
            "loss": sum(terms.values()).detach(),
            "l2_error": err,
            **{f"loss_{k}": v.detach() for k, v in terms.items()},
        }

    def configure_optimizers(self) -> OptimSpec:
        opt = (
            self.optim(self.parameters())
            if self.optim is not None
            else torch.optim.AdamW(self.parameters(), lr=self.lr)
        )
        sched = self.sched(opt) if self.sched is not None else None
        return OptimSpec.of(opt, sched, interval="step")
