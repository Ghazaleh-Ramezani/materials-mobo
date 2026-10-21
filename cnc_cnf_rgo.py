"""CNC/CNF/rGO film design space and a Lasso surrogate objective.

`LassoSurrogateObjective` fits one Lasso model per objective to the
formulations in `data/cnc_cnf_rgo_data.csv` and stands in for the lab: the
optimizer proposes a formulation, the surrogate returns conductivity and
tensile strength. A measured objective has the same call signature (a batch
of 7-parameter formulations in, a batch of objective values out), so it drops
into `run_mobo` unchanged.
"""

from __future__ import annotations

import pandas as pd
import torch
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

INPUT_COLUMNS = ["CNC", "CNF", "rGO", "acid_conc", "pH", "temperature", "time"]
OUTPUT_COLUMNS = ["conductivity", "tensile", "thickness"]
INPUT_BOUNDS = torch.tensor(
    [[0.1, 0.1, 0.05, 0.05, 5.0, 80.0, 3.0], [1.0, 0.8, 0.20, 0.50, 6.0, 95.0, 6.0]],
    dtype=torch.double,
)


def load_dataframe(csv_path):
    df = pd.read_csv(csv_path, comment="#")
    df.columns = df.columns.str.strip()
    return df


class LassoSurrogateObjective:
    def __init__(self, csv_path, agent="lascorbic", objectives=("conductivity", "tensile"), alpha=0.01):
        self.objectives = list(objectives)
        df = load_dataframe(csv_path)
        df = df[df["agent"].str.lower().str.startswith(agent[:3].lower())].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"No rows for agent {agent!r} in {csv_path}")
        self.n_rows = len(df)
        X = df[INPUT_COLUMNS].to_numpy(dtype=float)
        self._scaler = StandardScaler().fit(X)
        Xs = self._scaler.transform(X)
        self._models = {}
        for obj in self.objectives:
            y = df[obj].to_numpy(dtype=float)
            self._models[obj] = Lasso(alpha=alpha, max_iter=5000).fit(Xs, y)

    def coefficients(self) -> dict[str, list[float]]:
        """Standardised-input coefficients, one list per objective, in
        INPUT_COLUMNS order."""
        return {obj: m.coef_.tolist() for obj, m in self._models.items()}

    def ref_point(self, x, pad=0.15):
        """Reference point just below the objective values reached at `x`."""
        y0 = self(x)
        lo = y0.min(dim=0).values
        return lo - pad * (y0.max(dim=0).values - lo).clamp_min(1e-9)

    def __call__(self, x):
        xs = self._scaler.transform(x.detach().cpu().double().numpy())
        cols = [torch.tensor(self._models[obj].predict(xs), dtype=torch.double) for obj in self.objectives]
        return torch.stack(cols, dim=-1)
