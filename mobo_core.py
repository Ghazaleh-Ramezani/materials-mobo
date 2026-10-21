"""MOBO engine: independent GPs per objective, qLogEHVI acquisition, and a
weighted pick from the Pareto front for the human-in-the-loop step.

qLogEHVI integrates the hypervolume improvement over the region *not yet*
dominated by the observed front, so it takes a box decomposition of the
non-dominated region (`FastNondominatedPartitioning`). `DominatedPartitioning`
decomposes the opposite region; it is what computes the hypervolume itself,
and it is used for that below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.model_list_gp_regression import ModelListGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.box_decompositions.non_dominated import (
    FastNondominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls import SumMarginalLogLikelihood

torch.set_default_dtype(torch.double)


@dataclass
class MOBOConfig:
    bounds: torch.Tensor
    ref_point: torch.Tensor
    n_initial: int = 8
    n_iterations: int = 12
    batch_size: int = 1
    mc_samples: int = 128
    num_restarts: int = 10
    raw_samples: int = 256
    seed: int = 0
    verbose: bool = True

    @property
    def dim(self) -> int:
        return self.bounds.shape[1]

    @property
    def num_objectives(self) -> int:
        return self.ref_point.numel()


@dataclass
class MOBOHistory:
    train_x: torch.Tensor
    train_y: torch.Tensor
    hypervolume: list = field(default_factory=list)
    iter_index: list = field(default_factory=list)

    def pareto_mask(self) -> torch.Tensor:
        return is_non_dominated(self.train_y)

    def pareto_points(self) -> tuple[torch.Tensor, torch.Tensor]:
        mask = self.pareto_mask()
        return self.train_x[mask], self.train_y[mask]


def hypervolume(train_y: torch.Tensor, ref_point: torch.Tensor) -> float:
    """Hypervolume dominated by `train_y` and bounded below by `ref_point`."""
    return DominatedPartitioning(ref_point=ref_point, Y=train_y).compute_hypervolume().item()


def initial_design(bounds: torch.Tensor, n: int, seed: int) -> torch.Tensor:
    return draw_sobol_samples(bounds=bounds, n=n, q=1, seed=seed).squeeze(1)


def _build_model(train_x_norm: torch.Tensor, train_y: torch.Tensor) -> ModelListGP:
    models = [
        SingleTaskGP(train_x_norm, train_y[:, i : i + 1], outcome_transform=Standardize(m=1))
        for i in range(train_y.shape[-1])
    ]
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model


def build_acquisition(
    model, train_y: torch.Tensor, ref_point: torch.Tensor, mc_samples: int = 128
) -> qLogExpectedHypervolumeImprovement:
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([mc_samples]))
    partitioning = FastNondominatedPartitioning(ref_point=ref_point, Y=train_y)
    return qLogExpectedHypervolumeImprovement(
        model=model, ref_point=ref_point.tolist(), partitioning=partitioning, sampler=sampler
    )


def _propose(model, train_y: torch.Tensor, cfg: MOBOConfig) -> torch.Tensor:
    acqf = build_acquisition(model, train_y, cfg.ref_point, cfg.mc_samples)
    std_bounds = torch.zeros(2, cfg.dim)
    std_bounds[1] = 1.0
    candidate_norm, _ = optimize_acqf(
        acq_function=acqf,
        bounds=std_bounds,
        q=cfg.batch_size,
        num_restarts=cfg.num_restarts,
        raw_samples=cfg.raw_samples,
        sequential=True,
    )
    return candidate_norm


def run_mobo(objective, cfg: MOBOConfig, initial_x=None, initial_y=None) -> MOBOHistory:
    """Run the loop. Pass `initial_y` with `initial_x` to start from an
    already-evaluated design, e.g. one shared with a baseline."""
    torch.manual_seed(cfg.seed)
    if initial_x is None:
        initial_x = initial_design(cfg.bounds, cfg.n_initial, cfg.seed)
    train_x = initial_x
    train_y = objective(train_x) if initial_y is None else initial_y
    history = MOBOHistory(train_x=train_x.clone(), train_y=train_y.clone())
    hv = hypervolume(train_y, cfg.ref_point)
    history.hypervolume.append(hv)
    history.iter_index.append(0)
    if cfg.verbose:
        print(f"[init] n={train_x.shape[0]:>3d} hypervolume={hv:.4f}")
    for it in range(1, cfg.n_iterations + 1):
        model = _build_model(normalize(train_x, cfg.bounds), train_y)
        candidate_x = unnormalize(_propose(model, train_y, cfg), cfg.bounds)
        candidate_y = objective(candidate_x)
        train_x = torch.cat([train_x, candidate_x])
        train_y = torch.cat([train_y, candidate_y])
        hv = hypervolume(train_y, cfg.ref_point)
        history.train_x = train_x.clone()
        history.train_y = train_y.clone()
        history.hypervolume.append(hv)
        history.iter_index.append(it)
        if cfg.verbose:
            print(
                f"[iter {it:>2d}] n={train_x.shape[0]:>3d} hypervolume={hv:.4f} "
                f"pareto={int(is_non_dominated(train_y).sum())}"
            )
    return history


def run_random(objective, cfg: MOBOConfig, initial_x, initial_y) -> list[float]:
    """Uniform random search from the same evaluated initial design, same
    budget. Returns the hypervolume after each evaluation, aligned with
    MOBOHistory.hypervolume."""
    gen = torch.Generator().manual_seed(10_000 + cfg.seed)
    n_new = cfg.n_iterations * cfg.batch_size
    u = torch.rand(n_new, cfg.dim, generator=gen)
    x_new = unnormalize(u, cfg.bounds)
    torch.manual_seed(20_000 + cfg.seed)  # observation noise, if the objective has any
    y_new = objective(x_new)
    y = torch.cat([initial_y, y_new])
    n0 = initial_y.shape[0]
    return [hypervolume(y[: n0 + k * cfg.batch_size], cfg.ref_point) for k in range(cfg.n_iterations + 1)]


def select_with_preference(pareto_y, pareto_x, weights):
    y = pareto_y.clone()
    span = (y.max(dim=0).values - y.min(dim=0).values).clamp_min(1e-9)
    y_norm = (y - y.min(dim=0).values) / span
    w = weights / weights.sum()
    best = int(torch.argmax((y_norm * w).sum(dim=1)))
    return pareto_x[best], pareto_y[best], best
