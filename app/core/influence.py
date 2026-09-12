"""影响系数矩阵辨识。

模型：v = v0 + α·w
  v0 : 基线复振动向量 (n_sensors,)
  α  : 影响系数矩阵 (n_sensors, n_planes)，单位 振动/克
  w  : 校正面复配重向量 (n_planes,)

对每次试重运行取与基线的差分 Δv_k = v_k - v0，堆叠为
  ΔV (n_runs, n_sensors) = T (n_runs, n_planes) @ αᵀ
最小二乘解出 α，并给出逐运行逐测点的拟合残差。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..errors import IllConditionedMatrix, InsufficientTrials

#: 试重矩阵（列归一化后）条件数上限，超过即判病态
DEFAULT_CONDITION_THRESHOLD = 1.0e3


@dataclass
class InfluenceResult:
    coefficients: np.ndarray          # (n_sensors, n_planes) 复数
    residuals: np.ndarray             # (n_runs, n_sensors) 复数拟合残差
    relative_residuals: np.ndarray    # (n_runs, n_sensors) 残差/|Δv|
    condition: float                  # 列归一化试重矩阵条件数
    run_ids: list[int] = field(default_factory=list)


def fit_influence_coefficients(
    baseline: np.ndarray,
    trial_vectors: np.ndarray,
    trial_weights: np.ndarray,
    run_ids: list[int] | None = None,
    plane_names: list[str] | None = None,
    condition_threshold: float = DEFAULT_CONDITION_THRESHOLD,
) -> InfluenceResult:
    """拟合影响系数矩阵。

    :param baseline: (n_sensors,) 基线复振动
    :param trial_vectors: (n_runs, n_sensors) 各试重运行的复振动
    :param trial_weights: (n_runs, n_planes) 各运行施加在各面的复试重
    :raises InsufficientTrials: 无试重、某面无试重或试重矩阵秩亏
    :raises IllConditionedMatrix: 试重矩阵病态
    """
    baseline = np.asarray(baseline, dtype=complex)
    T = np.asarray(trial_weights, dtype=complex)
    if T.ndim != 2 or T.shape[0] == 0:
        raise InsufficientTrials(
            "没有任何试重运行，无法辨识影响系数",
            details={"missing": "trial_runs"},
        )
    n_runs, n_planes = T.shape
    names = plane_names or [f"plane_{p}" for p in range(n_planes)]

    # 每个面至少有一次非零试重
    empty_planes = [names[p] for p in range(n_planes) if not np.any(np.abs(T[:, p]) > 0)]
    if empty_planes:
        raise InsufficientTrials(
            "以下校正面缺少试重运行: " + ", ".join(empty_planes),
            details={"planes_without_trials": empty_planes},
        )

    # 秩检查：试重向量须张成全部校正面
    rank = int(np.linalg.matrix_rank(T))
    if rank < n_planes:
        raise InsufficientTrials(
            f"试重矩阵秩不足（rank={rank} < {n_planes}）：各次试重不独立，"
            "请在不同面上或不同角度施加试重",
            details={"rank": rank, "required": n_planes,
                     "run_ids": run_ids or []},
        )

    delta_v = np.asarray(trial_vectors, dtype=complex) - baseline  # (n_runs, n_sensors)

    coef_t, _, _, _ = np.linalg.lstsq(T, delta_v, rcond=None)  # (n_planes, n_sensors)
    coefficients = coef_t.T                                     # (n_sensors, n_planes)

    fitted = T @ coef_t
    residuals = delta_v - fitted
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(
            np.abs(delta_v) > 0,
            np.abs(residuals) / np.maximum(np.abs(delta_v), 1e-12),
            0.0,
        )

    # 条件数按列归一化计算，避免质量量级差异造成误判
    col_norms = np.linalg.norm(T, axis=0)
    Tn = T / col_norms
    condition = float(np.linalg.cond(Tn))
    if not np.isfinite(condition) or condition > condition_threshold:
        raise IllConditionedMatrix(
            f"试重矩阵病态（条件数 {condition:.3g} > {condition_threshold:g}）："
            "各次试重过于相似，请增大试重差异（质量或角度）",
            details={"condition": condition,
                     "threshold": condition_threshold,
                     "run_ids": run_ids or []},
        )

    return InfluenceResult(
        coefficients=coefficients,
        residuals=residuals,
        relative_residuals=relative,
        condition=condition,
        run_ids=run_ids or [],
    )
