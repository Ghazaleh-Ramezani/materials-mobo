
import dataclasses
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
from botorch.utils.multi_objective.box_decompositions.dominated import DominatedPartitioning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.sampling import draw_sobol_samples
from mobo_core import MOBOConfig, run_mobo, select_with_preference
from objectives import SPM_BOUNDS, SPM_REF_POINT, spm_imaging_objective
RESULTS = Path(__file__).parent/"results"; RESULTS.mkdir(exist_ok=True)
OBJ_NAMES = ["Quality","Reproducibility","Efficiency"]

def random_baseline(cfg):
    total = cfg.n_initial+cfg.n_iterations*cfg.batch_size
    x = draw_sobol_samples(bounds=cfg.bounds,n=total,q=1,seed=cfg.seed+1).squeeze(1)
    y = spm_imaging_objective(x)
    return [DominatedPartitioning(ref_point=cfg.ref_point,Y=y[:k]).compute_hypervolume().item()
            for k in range(cfg.n_initial,total+1)]

def main():
    cfg = MOBOConfig(bounds=SPM_BOUNDS,ref_point=SPM_REF_POINT,n_initial=10,n_iterations=18,seed=0)
    print("=== MOBO vs random (3 seeds) ===")
    mobo_curves,rand_curves,last_hist=[],[],None
    for s in range(3):
        c = dataclasses.replace(cfg,seed=s,verbose=(s==0))
        hist = run_mobo(spm_imaging_objective,c)
        mobo_curves.append(hist.hypervolume); rand_curves.append(random_baseline(c)); last_hist=hist
        print(f"  seed {s}: MOBO={hist.hypervolume[-1]:.4f}  random={rand_curves[-1][-1]:.4f}")
    mc,rc = torch.tensor(mobo_curves),torch.tensor(rand_curves)
    n_eval = [cfg.n_initial+i for i in range(mc.shape[1])]
    fig,ax = plt.subplots(figsize=(7,4.5))
    ax.plot(n_eval,mc.mean(0),"o-",lw=2,label="MOBO (qLogEHVI)")
    ax.fill_between(n_eval,mc.mean(0)-mc.std(0),mc.mean(0)+mc.std(0),alpha=0.2)
    ax.plot(list(range(cfg.n_initial,cfg.n_initial+rc.shape[1])),rc.mean(0),"s--",lw=1.8,color="gray",label="Random")
    ax.fill_between(list(range(cfg.n_initial,cfg.n_initial+rc.shape[1])),rc.mean(0)-rc.std(0),rc.mean(0)+rc.std(0),alpha=0.15,color="gray")
    ax.set_xlabel("Experiments"); ax.set_ylabel("Hypervolume"); ax.set_title("MOBO vs Random"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(RESULTS/"hypervolume.png",dpi=140); plt.close(fig)
    all_y=last_hist.train_y; mask=last_hist.pareto_mask()
    fig,axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,(i,j) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(all_y[~mask,i],all_y[~mask,j],c="lightgray",s=30,label="dominated")
        ax.scatter(all_y[mask,i],all_y[mask,j],c="crimson",s=60,edgecolor="k",label="Pareto",zorder=3)
        ax.set_xlabel(OBJ_NAMES[i]); ax.set_ylabel(OBJ_NAMES[j]); ax.grid(alpha=0.3)
    axes[0].legend(); fig.suptitle("Pareto front"); fig.tight_layout()
    fig.savefig(RESULTS/"pareto_front.png",dpi=140); plt.close(fig)
    print("Done! See results/")

if __name__=="__main__": main()
