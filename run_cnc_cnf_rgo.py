
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from cnc_cnf_rgo import INPUT_BOUNDS, INPUT_COLUMNS, LassoSurrogateObjective, feasible_initial_design
from mobo_core import MOBOConfig, run_mobo, select_with_preference

RESULTS = Path(__file__).parent/"results_cnc"; RESULTS.mkdir(exist_ok=True)
DATA = Path(__file__).parent/"data"/"cnc_cnf_rgo_data.csv"

def main():
    obj = LassoSurrogateObjective(DATA, agent="lascorbic", objectives=("conductivity","tensile"))
    init_x = feasible_initial_design(n=5, seed=0)
    ref_point = obj.ref_point(init_x)
    cfg = MOBOConfig(bounds=INPUT_BOUNDS, ref_point=ref_point, n_initial=5, n_iterations=10, seed=0)
    hist = run_mobo(obj, cfg, initial_x=init_x)
    pareto_x,pareto_y = hist.pareto_points()
    x_sel,y_sel,_ = select_with_preference(pareto_y,pareto_x,torch.tensor([2.0,1.0]))
    print("Best pick:")
    for n,v in zip(INPUT_COLUMNS,x_sel.tolist()): print(f"  {n:>12s} = {v:.3f}")
    print(f"  -> conductivity={y_sel[0]:.3f}  tensile={y_sel[1]:.2f} MPa")
    fig,ax=plt.subplots(figsize=(7,5))
    all_y=hist.train_y; mask=hist.pareto_mask()
    ax.scatter(all_y[~mask,0],all_y[~mask,1],c="lightgray",s=35,label="explored")
    ax.scatter(all_y[mask,0],all_y[mask,1],c="crimson",s=70,edgecolor="k",zorder=3,label="Pareto")
    ax.scatter(y_sel[0],y_sel[1],marker="*",s=320,c="gold",edgecolor="k",zorder=4,label="best pick")
    ax.set_title("CNC/CNF/rGO Pareto front")
    ax.set_xlabel("Conductivity (S/m)"); ax.set_ylabel("Tensile (MPa)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(RESULTS/"pareto_cnc_cnf_rgo.png",dpi=140); plt.close(fig)
    print("Wrote results_cnc/")

if __name__=="__main__": main()
