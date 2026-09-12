"""连续（理想）校正配重计算。

求解 min_w ||α·w + v0||₂，s.t. 每个校正面的合成配重质量 |w_p| ≤ M_p。
采用无约束最小二乘 + 违限面逐次投影固定：把超限面钳制到质量上限
（保持角度），固定后对其余面重新求解。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ContinuousSolution:
    weights: np.ndarray               # (n_planes,) 复配重（g·∠°）
    predicted_residual: np.ndarray    # (n_sensors,) 预测残余复振动
    limited_planes: list[int]         # 被质量上限钳制的面下标


def solve_continuous(
    alpha: np.ndarray,
    v0: np.ndarray,
    mass_limits: list[float] | np.ndarray,
) -> ContinuousSolution:
    alpha = np.asarray(alpha, dtype=complex)
    v0 = np.asarray(v0, dtype=complex)
    limits = np.asarray(mass_limits, dtype=float)
    n_planes = alpha.shape[1]

    w = np.zeros(n_planes, dtype=complex)
    fixed: dict[int, complex] = {}
    limited: list[int] = []
    free = list(range(n_planes))

    while free:
        A_free = alpha[:, free]
        # 已固定面的贡献并入目标
        residual_target = v0 + alpha[:, list(fixed)] @ np.array(list(fixed.values())) if fixed else v0
        sol, *_ = np.linalg.lstsq(A_free, -residual_target, rcond=None)

        # 找超限最严重的自由面
        worst_idx, worst_ratio = None, 1.0
        for i, p in enumerate(free):
            mag = abs(sol[i])
            if limits[p] > 0 and mag > limits[p] * worst_ratio:
                worst_idx, worst_ratio = i, mag / limits[p]
            elif limits[p] <= 0 and mag > 0:
                worst_idx, worst_ratio = i, np.inf

        if worst_idx is None:
            for i, p in enumerate(free):
                w[p] = sol[i]
            break

        # 钳制该面：保持角度，幅值取上限
        p = free.pop(worst_idx)
        mag = abs(sol[worst_idx])
        w[p] = limits[p] * np.exp(1j * np.angle(sol[worst_idx])) if mag > 0 else 0j
        fixed[p] = w[p]
        limited.append(p)

    predicted = v0 + alpha @ w
    return ContinuousSolution(weights=w, predicted_residual=predicted, limited_planes=limited)
