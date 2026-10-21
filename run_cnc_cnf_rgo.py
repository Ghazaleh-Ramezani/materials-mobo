"""MOBO on the CNC/CNF/rGO surrogate, against random search, with a weighted
pick from the final Pareto front.

The reference point is fixed once from a dense Sobol sample of the design
space, so hypervolumes are comparable across seeds and arms. Writes
results_cnc/summary.json and results_cnc/pareto_cnc_cnf_rgo.svg.

    python run_cnc_cnf_rgo.py --seeds 5
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "materials-mobo"})
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from cnc_cnf_rgo import INPUT_BOUNDS, INPUT_COLUMNS, LassoSurrogateObjective  # noqa: E402
from mobo_core import (  # noqa: E402
    MOBOConfig,
    initial_design,
    run_mobo,
    run_random,
    select_with_preference,
)

RESULTS = Path(__file__).parent / "results_cnc"
DATA = Path(__file__).parent / "data" / "cnc_cnf_rgo_data.csv"
WEIGHTS = torch.tensor([2.0, 1.0])  # conductivity weighted twice tensile strength


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-initial", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    obj = LassoSurrogateObjective(DATA, agent="lascorbic", objectives=("conductivity", "tensile"))
    ref_point = obj.ref_point(initial_design(INPUT_BOUNDS, 1024, seed=1234))
    cfg = MOBOConfig(
        bounds=INPUT_BOUNDS,
        ref_point=ref_point,
        n_initial=args.n_initial,
        n_iterations=args.iterations,
    )

    mobo_curves, rand_curves, hist0 = [], [], None
    for s in range(args.seeds):
        c = dataclasses.replace(cfg, seed=s, verbose=False)
        x0 = initial_design(INPUT_BOUNDS, c.n_initial, s)
        y0 = obj(x0)
        hist = run_mobo(obj, c, initial_x=x0, initial_y=y0)
        rand = run_random(obj, c, x0, y0)
        mobo_curves.append(hist.hypervolume)
        rand_curves.append(rand)
        if s == 0:
            hist0 = hist
        print(f"seed {s}: MOBO={hist.hypervolume[-1]:.3f}  random={rand[-1]:.3f}", flush=True)

    mc, rc = torch.tensor(mobo_curves), torch.tensor(rand_curves)
    pareto_x, pareto_y = hist0.pareto_points()
    x_sel, y_sel, _ = select_with_preference(pareto_y, pareto_x, WEIGHTS)
    summary = {
        "surrogate": {
            "rows": obj.n_rows,
            "inputs": INPUT_COLUMNS,
            "lasso_coefficients_standardised": obj.coefficients(),
        },
        "config": {
            "n_initial": cfg.n_initial,
            "n_iterations": cfg.n_iterations,
            "ref_point": ref_point.tolist(),
        },
        "n_seeds": args.seeds,
        "mobo_final_hv_mean": mc[:, -1].mean().item(),
        "mobo_final_hv_std": mc[:, -1].std().item(),
        "random_final_hv_mean": rc[:, -1].mean().item(),
        "random_final_hv_std": rc[:, -1].std().item(),
        "seeds_mobo_ahead": int((mc[:, -1] > rc[:, -1]).sum()),
        "seed0_pareto_size": int(pareto_x.shape[0]),
        "seed0_pick": {
            "weights": WEIGHTS.tolist(),
            "formulation": dict(zip(INPUT_COLUMNS, x_sel.tolist())),
            "conductivity_S_per_m": y_sel[0].item(),
            "tensile_MPa": y_sel[1].item(),
        },
        "mobo_curves": mobo_curves,
        "random_curves": rand_curves,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print("Seed-0 pick (conductivity weighted 2:1):")
    for n, v in zip(INPUT_COLUMNS, x_sel.tolist()):
        print(f"  {n:>12s} = {v:.3f}")
    print(f"  -> conductivity={y_sel[0]:.3f} S/m  tensile={y_sel[1]:.2f} MPa")

    fig, ax = plt.subplots(figsize=(7, 5))
    all_y, mask = hist0.train_y, hist0.pareto_mask()
    ax.scatter(all_y[~mask, 0], all_y[~mask, 1], c="lightgray", s=35, label="explored")
    ax.scatter(all_y[mask, 0], all_y[mask, 1], c="crimson", s=70, edgecolor="k", zorder=3, label="Pareto")
    ax.scatter(
        y_sel[0], y_sel[1], marker="*", s=320, c="gold", edgecolor="k", zorder=4, label="weighted pick"
    )
    ax.set_title("CNC/CNF/rGO Pareto front (Lasso surrogate, seed 0)")
    ax.set_xlabel("Conductivity (S/m)")
    ax.set_ylabel("Tensile strength (MPa)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "pareto_cnc_cnf_rgo.svg", metadata={"Date": None})
    plt.close(fig)
    print("wrote results_cnc/summary.json, results_cnc/pareto_cnc_cnf_rgo.svg")


if __name__ == "__main__":
    main()
