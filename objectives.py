"""Synthetic 3-objective benchmark modelled on scanning-probe-microscope
imaging, the setting Liu & Kalinin (2025) use for adaptive MOBO.

Five normalised controls (setpoint, proportional gain, scan rate, integral
gain, drive amplitude) map to image quality, reproducibility and acquisition
efficiency, all maximised. Quality peaks at interior gains; efficiency rewards
fast scans that quality penalises, so the three trade off and the Pareto front
is a surface rather than a point.
"""

from __future__ import annotations

import torch

SPM_BOUNDS = torch.tensor([[0.0] * 5, [1.0] * 5])
SPM_REF_POINT = torch.tensor([-0.5, -0.5, -0.5])


def spm_imaging_objective(x, noise_std=0.02):
    setpoint, p_gain, scan_rate, i_gain, drive = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
    rq = (
        1.4
        * torch.exp(-((setpoint - 0.6) ** 2) / 0.12)
        * torch.exp(-((p_gain - 0.65) ** 2) / 0.14)
        * torch.exp(-((i_gain - 0.55) ** 2) / 0.16)
        - 0.15 * scan_rate**2
        - 0.10 * (drive - 0.5) ** 2
    )
    rr = (
        0.7 * (1.0 - p_gain) ** 1.5
        + 0.5 * (1.0 - i_gain)
        + 0.5 * (1.0 - scan_rate)
        - 0.4 * (drive - 0.3) ** 2
    ) / 1.7
    re = 0.6 * scan_rate + 0.2 * (1.0 - setpoint) + 0.2 * (1.0 - drive)
    y = torch.stack([rq, rr, re], dim=-1)
    if noise_std > 0:
        y = y + noise_std * torch.randn_like(y)
    return y
