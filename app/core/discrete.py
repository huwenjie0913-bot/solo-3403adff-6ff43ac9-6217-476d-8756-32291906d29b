"""离散可安装配重组合搜索与公差最差情形评估。

每个校正面：在固定孔位角度上放置配重规格表中的配重（每孔至多一块，
每面至多 max_weights 块，总安装质量不超过该面上限），枚举得到候选
合成向量；按与连续解目标的接近程度剪枝后，对两面候选做笛卡尔组合，
用预测残振、总配重、公差最差结果三个指标排序。

最差情形模型（一阶三角不等式界，保守）：
  wc_s = |pred_s|
       + Σ_p |α_sp| · 2·sin(δ/2) · M_p        （安装角公差 δ 引起的配重向量偏差）
       + A0_s · (ε_a + 2·sin(ε_φ/2))           （基线幅值相对误差 ε_a 与相位误差 ε_φ）
其中 M_p 为 p 面总安装质量，A0_s 为基线振幅。测量误差对 α 本身的
二阶影响忽略。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import numpy as np

from ..errors import NoFeasibleCombination

#: 单面枚举规模上限，防止孔位×规格×块数组合爆炸
ENUMERATION_CAP = 250_000


@dataclass
class WeightAssignment:
    plane: str
    hole_angle: float   # 孔位角度（度）
    mass: float         # 配重质量（g）


@dataclass
class PlaneCandidate:
    vector: complex                       # 合成复配重（g·∠°，半径已折入系数）
    assignments: list[WeightAssignment] = field(default_factory=list)
    total_mass: float = 0.0


@dataclass
class DiscreteSolution:
    weights: np.ndarray                   # (n_planes,) 合成复配重
    assignments: list[WeightAssignment]
    total_mass: float
    predicted_residual: np.ndarray        # (n_sensors,)
    predicted_metric: float               # max_s |pred_s|
    worst_case: float                     # 公差下最差残振指标


def enumerate_plane_candidates(
    plane_name: str,
    hole_angles: list[float],
    weight_specs: list[float],
    mass_limit: float,
    max_weights: int = 3,
) -> list[PlaneCandidate]:
    """枚举单面可安装组合（含空组合）。总安装质量 ≤ mass_limit。"""
    candidates = [PlaneCandidate(vector=0j)]
    holes = [float(a) % 360.0 for a in hole_angles]
    specs = [float(m) for m in weight_specs if m > 0]
    if not holes or not specs or mass_limit <= 0:
        return candidates

    for k in range(1, max_weights + 1):
        # 组合数护栏
        n_combos = _count_combos(len(holes), k) * len(specs) ** k
        if n_combos > ENUMERATION_CAP:
            break
        for hole_idx in combinations(range(len(holes)), k):
            angles = [holes[i] for i in hole_idx]
            for masses in product(specs, repeat=k):
                total = sum(masses)
                if total > mass_limit + 1e-9:
                    continue
                vec = sum(
                    m * np.exp(1j * np.deg2rad(a)) for m, a in zip(masses, angles)
                )
                candidates.append(
                    PlaneCandidate(
                        vector=complex(vec),
                        assignments=[
                            WeightAssignment(plane=plane_name, hole_angle=a, mass=m)
                            for a, m in zip(angles, masses)
                        ],
                        total_mass=total,
                    )
                )
    return candidates


def _count_combos(n: int, k: int) -> int:
    from math import comb
    return comb(n, k)


def prune_candidates(
    candidates: list[PlaneCandidate],
    target: complex,
    keep: int = 120,
) -> list[PlaneCandidate]:
    """按与连续目标向量的接近程度保留前 keep 个，始终保留空组合。"""
    zero = [c for c in candidates if not c.assignments]
    rest = sorted(
        (c for c in candidates if c.assignments),
        key=lambda c: (abs(c.vector - target), c.total_mass),
    )
    return zero + rest[:keep]


def worst_case_residual(
    predicted: np.ndarray,
    alpha: np.ndarray,
    installed_masses: list[float],
    baseline_amplitudes: np.ndarray,
    amp_error: float,
    phase_error_deg: float,
    angle_tolerance_deg: float,
) -> float:
    """公差下最差残振指标（各测点取最大）。"""
    pred = np.abs(np.asarray(predicted, dtype=complex))
    a = np.abs(np.asarray(alpha, dtype=complex))          # (n_sensors, n_planes)
    delta = np.deg2rad(angle_tolerance_deg)
    mount = a @ np.array([2.0 * np.sin(delta / 2.0) * m for m in installed_masses])
    meas = np.asarray(baseline_amplitudes, dtype=float) * (
        amp_error + 2.0 * np.sin(np.deg2rad(phase_error_deg) / 2.0)
    )
    return float(np.max(pred + mount + meas))


def search_discrete_solutions(
    alpha: np.ndarray,
    v0: np.ndarray,
    continuous_target: np.ndarray,
    plane_names: list[str],
    hole_angles: list[list[float]],
    weight_specs: list[float],
    mass_limits: list[float],
    amp_error: float,
    phase_error_deg: float,
    angle_tolerance_deg: float,
    max_weights_per_plane: int = 3,
    keep_per_plane: int = 120,
    top: int = 10,
) -> list[DiscreteSolution]:
    """搜索两面离散组合，按 (预测残振, 总配重, 最差结果) 排序返回前 top 个。"""
    n_planes = alpha.shape[1]
    per_plane: list[list[PlaneCandidate]] = []
    for p in range(n_planes):
        cands = enumerate_plane_candidates(
            plane_names[p], hole_angles[p], weight_specs,
            mass_limits[p], max_weights_per_plane,
        )
        installable = [c for c in cands if c.assignments]
        if not installable and abs(continuous_target[p]) > 1e-9:
            # 该面需要校正但无任何可安装组合：指出约束来源
            min_spec = min(weight_specs) if weight_specs else None
            raise NoFeasibleCombination(
                f"校正面 {plane_names[p]} 无可安装的配重组合",
                details={
                    "plane": plane_names[p],
                    "mass_limit": mass_limits[p],
                    "min_weight_spec": min_spec,
                    "hole_count": len(hole_angles[p]),
                    "required_resultant_mass": abs(continuous_target[p]),
                    "hint": "质量上限低于最小配重规格或未配置孔位/配重规格",
                },
            )
        per_plane.append(prune_candidates(cands, continuous_target[p], keep_per_plane))

    baseline_amps = np.abs(v0)
    solutions: list[DiscreteSolution] = []
    # 当前实现针对双面平衡（需求即双面）；逐对组合评估
    for c0 in per_plane[0]:
        for c1 in per_plane[1]:
            w = np.array([c0.vector, c1.vector], dtype=complex)
            pred = v0 + alpha @ w
            assignments = c0.assignments + c1.assignments
            masses = [c0.total_mass, c1.total_mass]
            solutions.append(
                DiscreteSolution(
                    weights=w,
                    assignments=assignments,
                    total_mass=c0.total_mass + c1.total_mass,
                    predicted_residual=pred,
                    predicted_metric=float(np.max(np.abs(pred))),
                    worst_case=worst_case_residual(
                        pred, alpha, masses, baseline_amps,
                        amp_error, phase_error_deg, angle_tolerance_deg,
                    ),
                )
            )

    solutions.sort(key=lambda s: (round(s.predicted_metric, 6),
                                  round(s.total_mass, 6),
                                  round(s.worst_case, 6)))
    return solutions[:top]
