
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
from botorch.utils.multi_objective.pareto import is_non_dominated
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
    def dim(self): return self.bounds.shape[1]
    @property
    def num_objectives(self): return self.ref_point.numel()

@dataclass
class MOBOHistory:
    train_x: torch.Tensor
    train_y: torch.Tensor
    hypervolume: list = field(default_factory=list)
    iter_index: list = field(default_factory=list)
    def pareto_mask(self): return is_non_dominated(self.train_y)
    def pareto_points(self):
        mask = self.pareto_mask()
        return self.train_x[mask], self.train_y[mask]

def _build_model(train_x_norm, train_y):
    models = [SingleTaskGP(train_x_norm, train_y[:,i:i+1], outcome_transform=Standardize(m=1)) for i in range(train_y.shape[-1])]
    model = ModelListGP(*models)
    fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood, model))
    return model

def _compute_hypervolume(train_y, ref_point):
    return DominatedPartitioning(ref_point=ref_point, Y=train_y).compute_hypervolume().item()

def _propose(model, train_y, cfg):
    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([cfg.mc_samples]))
    acqf = qLogExpectedHypervolumeImprovement(model=model, ref_point=cfg.ref_point.tolist(),
        partitioning=DominatedPartitioning(ref_point=cfg.ref_point, Y=train_y), sampler=sampler)
    std_bounds = torch.zeros(2, cfg.dim); std_bounds[1] = 1.0
    candidate_norm, _ = optimize_acqf(acq_function=acqf, bounds=std_bounds, q=cfg.batch_size,
        num_restarts=cfg.num_restarts, raw_samples=cfg.raw_samples, sequential=True)
    return candidate_norm

def run_mobo(objective, cfg, initial_x=None):
    torch.manual_seed(cfg.seed)
    if initial_x is None:
        from botorch.utils.sampling import draw_sobol_samples
        initial_x = draw_sobol_samples(bounds=cfg.bounds, n=cfg.n_initial, q=1, seed=cfg.seed).squeeze(1)
    train_x = initial_x; train_y = objective(train_x)
    history = MOBOHistory(train_x=train_x.clone(), train_y=train_y.clone())
    hv = _compute_hypervolume(train_y, cfg.ref_point)
    history.hypervolume.append(hv); history.iter_index.append(0)
    if cfg.verbose: print(f"[init] n={train_x.shape[0]:>3d} hypervolume={hv:.4f}")
    for it in range(1, cfg.n_iterations + 1):
        model = _build_model(normalize(train_x, cfg.bounds), train_y)
        candidate_x = unnormalize(_propose(model, train_y, cfg), cfg.bounds)
        candidate_y = objective(candidate_x)
        train_x = torch.cat([train_x, candidate_x]); train_y = torch.cat([train_y, candidate_y])
        hv = _compute_hypervolume(train_y, cfg.ref_point)
        history.train_x = train_x.clone(); history.train_y = train_y.clone()
        history.hypervolume.append(hv); history.iter_index.append(it)
        if cfg.verbose:
            print(f"[iter {it:>2d}] n={train_x.shape[0]:>3d} hypervolume={hv:.4f} pareto={int(is_non_dominated(train_y).sum())}")
    return history

def select_with_preference(pareto_y, pareto_x, weights):
    y = pareto_y.clone()
    span = (y.max(dim=0).values - y.min(dim=0).values).clamp_min(1e-9)
    y_norm = (y - y.min(dim=0).values) / span
    w = weights / weights.sum()
    best = int(torch.argmax((y_norm * w).sum(dim=1)))
    return pareto_x[best], pareto_y[best], best
