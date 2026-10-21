"""Fast checks for the acquisition, hypervolume and helpers. CPU, seconds."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from botorch.utils.testing import MockModel, MockPosterior

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cnc_cnf_rgo import INPUT_BOUNDS, LassoSurrogateObjective  # noqa: E402
from mobo_core import (  # noqa: E402
    MOBOConfig,
    build_acquisition,
    hypervolume,
    initial_design,
    run_mobo,
    run_random,
    select_with_preference,
)
from objectives import SPM_BOUNDS, SPM_REF_POINT, spm_imaging_objective  # noqa: E402

DATA = Path(__file__).resolve().parents[1] / "data" / "cnc_cnf_rgo_data.csv"

# Two-point front with reference (0, 0): boxes [0,1]x[0,0.5] and [0,0.5]x[0,1]
# overlap on [0,0.5]^2, so the hypervolume is 0.5 + 0.5 - 0.25 = 0.75.
FRONT = torch.tensor([[1.0, 0.5], [0.5, 1.0]], dtype=torch.double)
REF = torch.tensor([0.0, 0.0], dtype=torch.double)


def _ehvi_at(point: list[float]) -> float:
    """Expected hypervolume improvement of a candidate whose posterior is a
    point mass at `point`."""
    s = torch.tensor([[point]], dtype=torch.double)
    model = MockModel(MockPosterior(mean=s, variance=torch.zeros_like(s), samples=s))
    acqf = build_acquisition(model, FRONT, REF, mc_samples=16)
    return acqf(torch.zeros(1, 1, 2, dtype=torch.double)).exp().item()


def test_hypervolume_of_known_front():
    assert hypervolume(FRONT, REF) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "point, expected",
    [
        ([2.0, 2.0], 4.0 - 0.75),  # dominates the whole front
        ([1.0, 1.0], 1.0 - 0.75),  # fills the notch between the two points
        ([0.25, 0.25], 0.0),  # already dominated: no improvement
        ([1.5, 0.25], 0.5 * 0.25),  # extends the front along objective 1 only
    ],
)
def test_acquisition_equals_hypervolume_improvement(point, expected):
    # Regression test: with DominatedPartitioning the acquisition scored
    # overlap with the dominated region instead of improvement over it, off
    # by 0.06 to 2.5 on these points. The 5e-3 tolerance covers qLogEHVI's
    # smooth fat-max approximation (temperature 1e-2), which shows where a
    # candidate shares a coordinate with the front, as (1, 1) does.
    assert _ehvi_at(point) == pytest.approx(expected, abs=5e-3)


def test_initial_design_is_inside_bounds_and_seeded():
    a = initial_design(INPUT_BOUNDS, 16, seed=3)
    b = initial_design(INPUT_BOUNDS, 16, seed=3)
    assert torch.equal(a, b)
    assert (a >= INPUT_BOUNDS[0]).all() and (a <= INPUT_BOUNDS[1]).all()


def test_preference_pick_follows_weights():
    y = torch.tensor([[1.0, 0.0], [0.6, 0.6], [0.0, 1.0]])
    x = torch.arange(3.0).unsqueeze(-1)
    assert select_with_preference(y, x, torch.tensor([1.0, 0.0]))[2] == 0
    assert select_with_preference(y, x, torch.tensor([0.0, 1.0]))[2] == 2
    assert select_with_preference(y, x, torch.tensor([1.0, 1.0]))[2] == 1


def test_surrogate_shapes_and_missing_agent(tmp_path):
    obj = LassoSurrogateObjective(DATA)
    y = obj(initial_design(INPUT_BOUNDS, 4, seed=0))
    assert y.shape == (4, 2)
    with pytest.raises(ValueError, match="No rows"):
        LassoSurrogateObjective(DATA, agent="hydrazine")


def test_loop_and_baseline_share_the_initial_design():
    cfg = MOBOConfig(
        bounds=SPM_BOUNDS, ref_point=SPM_REF_POINT, n_initial=6, n_iterations=2,
        num_restarts=2, raw_samples=32, mc_samples=16, verbose=False,
    )
    torch.manual_seed(0)
    x0 = initial_design(cfg.bounds, cfg.n_initial, 0)
    y0 = spm_imaging_objective(x0)
    hist = run_mobo(spm_imaging_objective, cfg, initial_x=x0, initial_y=y0)
    rand = run_random(spm_imaging_objective, cfg, x0, y0)
    assert len(hist.hypervolume) == len(rand) == cfg.n_iterations + 1
    assert hist.hypervolume[0] == rand[0]  # same starting point
    assert all(b >= a for a, b in zip(hist.hypervolume, hist.hypervolume[1:]))  # HV never drops
    assert hist.train_x.shape == (cfg.n_initial + cfg.n_iterations, 5)
