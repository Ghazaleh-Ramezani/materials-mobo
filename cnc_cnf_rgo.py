
from __future__ import annotations
import pandas as pd, torch
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

INPUT_COLUMNS = ["CNC","CNF","rGO","acid_conc","pH","temperature","time"]
OUTPUT_COLUMNS = ["conductivity","tensile","thickness"]
INPUT_BOUNDS = torch.tensor([[0.1,0.1,0.05,0.05,5.0,80.0,3.0],
                              [1.0,0.8,0.20,0.50,6.0,95.0,6.0]], dtype=torch.double)

def load_dataframe(csv_path):
    df = pd.read_csv(csv_path, comment="#")
    df.columns = df.columns.str.strip()
    return df

class LassoSurrogateObjective:
    def __init__(self, csv_path, agent="lascorbic", objectives=("conductivity","tensile")):
        self.objectives = list(objectives)
        df = load_dataframe(csv_path)
        df = df[df["agent"].str.lower().str.startswith(agent[:3].lower())].reset_index(drop=True)
        X = df[INPUT_COLUMNS].to_numpy(dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._models = {}
        for obj in self.objectives:
            y = df[obj].to_numpy(dtype=float)
            self._models[obj] = Lasso(alpha=0.01, max_iter=5000).fit(Xs, y)

    def ref_point(self, init_x, pad=0.15):
        y0 = self(init_x)
        lo = y0.min(dim=0).values
        return lo - pad*(y0.max(dim=0).values - lo).clamp_min(1e-9)

    def __call__(self, x):
        xs = self._scaler.transform(x.detach().cpu().double().numpy())
        cols = [torch.tensor(self._models[obj].predict(xs), dtype=torch.double)
                for obj in self.objectives]
        return torch.stack(cols, dim=-1)

def feasible_initial_design(n=10, seed=0):
    from botorch.utils.sampling import draw_sobol_samples
    x = draw_sobol_samples(bounds=INPUT_BOUNDS, n=n, q=1, seed=seed).squeeze(1).double()
    total = x[:,0]+x[:,1]+x[:,2]
    factor = (2.0/total.clamp_min(1e-6)).clamp_max(1.0)
    x[:,0]*=factor; x[:,1]*=factor; x[:,2]*=factor
    return x
