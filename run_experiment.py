"""MOBO vs random search on the synthetic 3-objective SPM benchmark.

Both arms start from the same evaluated Sobol design and get the same budget,
so each seed is a paired comparison. Writes results/summary.json (every curve
and the summary statistics behind the README table) and results/hypervolume.svg.

    python run_experiment.py --seeds 10
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "materials-mobo"})
import botorch  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from mobo_core import MOBOConfig, initial_design, run_mobo, run_random  # noqa: E402
from objectives import SPM_BOUNDS, SPM_REF_POINT, spm_imaging_objective  # noqa: E402

RESULTS = Path(__file__).parent / "results"


def evaluations_to_reach(curve: list[float], target: float, n_initial: int) -> int | None:
    """Total evaluations (initial design included) at which `curve` first
    reaches `target`; None if it never does."""
    for k, hv in enumerate(curve):
        if hv >= target:
            return n_initial + k
    return None


def summarise(mobo: torch.Tensor, rand: torch.Tensor, n_initial: int) -> dict:
    final_m, final_r = mobo[:, -1], rand[:, -1]
    diff = final_m - final_r
    target = final_r.mean().item()
    mean_curve = mobo.mean(0).tolist()
    per_seed = [evaluations_to_reach(c, target, n_initial) for c in mobo.tolist()]
    return {
        "mobo_final_hv_mean": final_m.mean().item(),
        "mobo_final_hv_std": final_m.std().item(),
        "random_final_hv_mean": final_r.mean().item(),
        "random_final_hv_std": final_r.std().item(),
        "paired_diff_mean": diff.mean().item(),
        "paired_diff_std": diff.std().item(),
        "seeds_mobo_ahead": int((diff > 0).sum()),
        "n_seeds": int(mobo.shape[0]),
        "budget_evaluations": n_initial + mobo.shape[1] - 1,
        "target_hv": target,
        "mean_curve_evaluations_to_target": evaluations_to_reach(mean_curve, target, n_initial),
        "per_seed_evaluations_to_target": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-initial", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=18)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    cfg = MOBOConfig(
        bounds=SPM_BOUNDS,
        ref_point=SPM_REF_POINT,
        n_initial=args.n_initial,
        n_iterations=args.iterations,
    )
    mobo_curves, rand_curves = [], []
    for s in range(args.seeds):
        c = dataclasses.replace(cfg, seed=s, verbose=False)
        torch.manual_seed(s)
        x0 = initial_design(c.bounds, c.n_initial, s)
        y0 = spm_imaging_objective(x0)
        hist = run_mobo(spm_imaging_objective, c, initial_x=x0, initial_y=y0)
        rand = run_random(spm_imaging_objective, c, x0, y0)
        mobo_curves.append(hist.hypervolume)
        rand_curves.append(rand)
        print(f"seed {s}: MOBO={hist.hypervolume[-1]:.4f}  random={rand[-1]:.4f}", flush=True)

    mc, rc = torch.tensor(mobo_curves), torch.tensor(rand_curves)
    summary = summarise(mc, rc, cfg.n_initial)
    print(json.dumps(summary, indent=2))
    out = {
        "benchmark": "spm_imaging_objective (5 inputs, 3 objectives, noise_std 0.02)",
        "config": {
            "n_initial": cfg.n_initial,
            "n_iterations": cfg.n_iterations,
            "batch_size": cfg.batch_size,
            "mc_samples": cfg.mc_samples,
            "num_restarts": cfg.num_restarts,
            "raw_samples": cfg.raw_samples,
            "ref_point": cfg.ref_point.tolist(),
        },
        "summary": summary,
        "mobo_curves": mobo_curves,
        "random_curves": rand_curves,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "botorch": botorch.__version__,
        },
    }
    (RESULTS / "summary.json").write_text(json.dumps(out, indent=2) + "\n")

    n_eval = [cfg.n_initial + i for i in range(mc.shape[1])]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for curves, label, style, color in [(mc, "MOBO (qLogEHVI)", "o-", "C0"), (rc, "Random", "s--", "gray")]:
        m, sd = curves.mean(0), curves.std(0)
        ax.plot(n_eval, m, style, lw=2, color=color, label=label)
        ax.fill_between(n_eval, m - sd, m + sd, alpha=0.18, color=color)
    ax.set_xlabel("Experiments")
    ax.set_ylabel("Hypervolume")
    ax.set_title(f"MOBO vs random, mean ± std over {args.seeds} seeds")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "hypervolume.svg", metadata={"Date": None})
    plt.close(fig)

    print("wrote results/summary.json, results/hypervolume.svg")


if __name__ == "__main__":
    main()
