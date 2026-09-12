"""加重与去料混合校正：候选枚举、联合求解、安全余量与公差最差值。

物理模型（影响系数法的离散动作扩展）：

* 加重动作：在加重孔角度 θ 处加装质量 m，等效复校正量 +m·exp(iθ)；
* 去料动作：在钻削孔角度 θ 处钻去质量 m（该方向上等效质量减少，
  产生的不平衡力与“在该方向加重”相反），等效复校正量 **−m·exp(iθ)**，
  即以反向复矢量参与联合求解。

单面合成校正向量
    w_p = Σ_add m·exp(iθ) − Σ_remove m·exp(iθ)
预测残振（净振动空间）
    pred = v0 + α @ w

每个校正面分别录入：可用加重孔、可钻削孔位（含当前厚度、每孔去料
上限、最小剩余厚度）、已有配重、加重质量上限、去料质量上限与单面
总改变量上限。候选须同时满足孔位、质量、厚度（去料深度换算）及
单面改变量限制，按 (预测残振, 总改变量, −最小安全余量) 排序。

最差情形沿用 discrete.worst_case_residual 的一阶三角不等式界，
安装质量项取该面“加重 + 去料”总改变量（两者方向误差只会相加）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from math import comb

import numpy as np

from ..errors import MixedCorrectionConfigError, NoFeasibleMixedCorrection
from .discrete import worst_case_residual
from .vibration import normalize_angle

#: 单面（加重或去料）枚举规模上限，防止孔位×规格×块数组合爆炸
ENUMERATION_CAP = 250_000

EPS = 1e-9


# ---------------------------------------------------------------- 输入/输出结构


@dataclass
class ExistingWeight:
    """已安装配重（仅占据该面孔位与加重质量余量，不参与本次校正）。"""

    hole_angle: float       # 度
    mass: float             # g


@dataclass
class DrillHole:
    """可钻削孔位：几何厚度与每孔去料上限。"""

    hole_angle: float                       # 度
    current_thickness: float                # 当前剩余厚度 mm
    removal_limit: float                    # 每孔去料质量上限 g
    min_remaining_thickness: float = 0.0    # 钻后最小允许剩余厚度 mm

    @property
    def thickness_limited_mass(self) -> float:
        """由最小剩余厚度决定的可去料质量（需由 mass_per_mm 换算后使用）。"""
        return max(self.current_thickness - self.min_remaining_thickness, 0.0)


@dataclass
class PlaneSpec:
    """单个校正面的混合校正几何与约束快照（纯数值，与面名解耦）。"""

    add_hole_angles: list[float]
    weight_specs: list[float]
    drill_holes: list[DrillHole]
    existing_weights: list[ExistingWeight]
    add_mass_limit: float               # 本次加重总质量上限 g
    remove_mass_limit: float            # 本次去料总质量上限 g
    change_mass_limit: float            # 单面总改变量（加重+去料）上限 g
    mass_per_mm: float | None = None    # 钻削深度→去料质量换算 g/mm
    max_weights_per_plane: int = 3

    def effective_hole_limit(self, h: DrillHole) -> float:
        """单孔可去料质量 = min(每孔去料上限, 厚度允许质量)。"""
        limit = float(h.removal_limit)
        if self.mass_per_mm is not None:
            limit = min(limit, h.thickness_limited_mass * self.mass_per_mm)
        return limit


@dataclass
class MixedAction:
    """一个可执行动作：加重或去料。"""

    kind: str               # add | remove
    hole_angle: float       # 度（归一化）
    mass: float             # g

    @property
    def vector(self) -> complex:
        sign = 1.0 if self.kind == "add" else -1.0
        return complex(sign * self.mass * np.exp(1j * np.deg2rad(self.hole_angle)))

    @property
    def change(self) -> float:
        return float(abs(self.mass))

    def as_json(self) -> dict:
        return {"kind": self.kind, "hole_angle": self.hole_angle, "mass": float(self.mass)}


@dataclass
class PlaneSelection:
    """单面选择：一组加重/去料动作及其合成向量与质量统计。"""

    actions: list[MixedAction]
    vector: complex
    add_mass: float
    remove_mass: float

    @property
    def change_mass(self) -> float:
        return self.add_mass + self.remove_mass


@dataclass
class EvaluatedAction:
    """单个动作的评估明细：对各测点的振动贡献、去料深度与厚度安全。"""

    kind: str
    hole_angle: float
    mass: float
    vector: complex
    plane: str = ""
    effects: dict[str, complex] = field(default_factory=dict)   # sensor -> α·w
    drill_depth: float | None = None        # mm，仅去料
    remaining_thickness: float | None = None
    thickness_margin: float | None = None   # 钻后剩余 − 最小剩余，mm
    thickness_margin_ratio: float | None = None


@dataclass
class MixedCandidate:
    """一个完整的双面混合校正候选及全部评估量。"""

    actions: list[MixedAction]
    weights: np.ndarray                     # (n_planes,) 合成复校正向量
    predicted_residual: np.ndarray          # (n_sensors,)
    predicted_metric: float
    total_change: float
    worst_case: float
    min_safety_margin: float                # 最小相对安全余量（≤0 即违反）
    critical_constraint: dict | None        # {plane, constraint, ...}
    plane_safety: list[dict]
    evaluated_actions: list[EvaluatedAction]
    plane_selections: list[PlaneSelection] = field(default_factory=list)


# ---------------------------------------------------------------- 规格构建与校验


def _angle_list(angles, field_name: str) -> list[float]:
    norm = [normalize_angle(float(a)) for a in angles]
    if len(norm) != len(set(round(a, 6) for a in norm)):
        raise MixedCorrectionConfigError(
            f"{field_name}存在重复孔位角度",
            {"field": field_name, "angles": norm},
        )
    return norm


def build_plane_spec(
    *,
    add_hole_angles: list[float],
    weight_specs: list[float],
    drill_holes: list[dict],
    existing_weights: list[dict],
    add_mass_limit: float,
    remove_mass_limit: float | None,
    change_mass_limit: float | None,
    mass_per_mm: float | None,
    max_weights_per_plane: int,
) -> PlaneSpec:
    """把服务层传入的面配置转成 PlaneSpec，并做面内自洽校验。

    面名、面集合归属由服务层校验；这里只校验数值与几何自洽性。
    """
    add_angles = _angle_list(add_hole_angles, "add_hole_angles")
    specs = sorted({float(m) for m in weight_specs})
    if not specs or any(m <= 0 for m in specs):
        raise MixedCorrectionConfigError(
            "配重规格必须为正数且至少一个",
            {"weight_specs": weight_specs},
        )

    holes: list[DrillHole] = []
    for h in drill_holes:
        dh = DrillHole(
            hole_angle=normalize_angle(float(h["hole_angle"])),
            current_thickness=float(h["current_thickness"]),
            removal_limit=float(h["removal_limit"]),
            min_remaining_thickness=float(h.get("min_remaining_thickness", 0.0)),
        )
        if dh.current_thickness < 0 or dh.min_remaining_thickness < 0:
            raise MixedCorrectionConfigError(
                "钻削孔厚度与最小剩余厚度不得为负",
                {"hole_angle": dh.hole_angle,
                 "current_thickness": dh.current_thickness,
                 "min_remaining_thickness": dh.min_remaining_thickness},
            )
        if dh.min_remaining_thickness > dh.current_thickness + 1e-9:
            raise MixedCorrectionConfigError(
                "钻削孔最小剩余厚度超过当前厚度，任何去料都不可行",
                {"hole_angle": dh.hole_angle,
                 "current_thickness": dh.current_thickness,
                 "min_remaining_thickness": dh.min_remaining_thickness},
            )
        if dh.removal_limit <= 0:
            raise MixedCorrectionConfigError(
                "每孔去料上限必须为正数",
                {"hole_angle": dh.hole_angle, "removal_limit": dh.removal_limit},
            )
        holes.append(dh)
    hole_angles = [h.hole_angle for h in holes]
    if len(hole_angles) != len(set(round(a, 6) for a in hole_angles)):
        raise MixedCorrectionConfigError(
            "可钻削孔位存在重复角度", {"hole_angles": hole_angles}
        )

    existing: list[ExistingWeight] = []
    occupied: set[float] = set()
    existing_total = 0.0
    for w in existing_weights:
        ew = ExistingWeight(
            hole_angle=normalize_angle(float(w["hole_angle"])),
            mass=float(w["mass"]),
        )
        if ew.mass < 0:
            raise MixedCorrectionConfigError(
                "已有配重质量不得为负",
                {"hole_angle": ew.hole_angle, "mass": ew.mass},
            )
        existing.append(ew)
        occupied.add(round(ew.hole_angle, 6))
        existing_total += ew.mass

    if mass_per_mm is not None:
        if mass_per_mm <= 0:
            raise MixedCorrectionConfigError(
                "去料质量/钻削深度换算必须为正数", {"mass_per_mm": mass_per_mm}
            )
    elif holes:
        raise MixedCorrectionConfigError(
            "配置了可钻削孔位但缺少去料质量/钻削深度换算 mass_per_mm",
            {"mass_per_mm": mass_per_mm},
        )

    if add_mass_limit is None or add_mass_limit < -EPS:
        raise MixedCorrectionConfigError(
            "加重质量上限缺失或为负", {"add_mass_limit": add_mass_limit}
        )

    # 默认去料上限 = Σ 单孔有效上限（含厚度换算）
    probe_spec = PlaneSpec(
        add_hole_angles=add_angles, weight_specs=specs, drill_holes=holes,
        existing_weights=existing, add_mass_limit=float(add_mass_limit),
        remove_mass_limit=0.0, change_mass_limit=0.0,
        mass_per_mm=mass_per_mm, max_weights_per_plane=max_weights_per_plane,
    )
    removable_capacity = sum(probe_spec.effective_hole_limit(h) for h in holes)
    r_limit = removable_capacity if remove_mass_limit is None else float(remove_mass_limit)
    if r_limit < -EPS:
        raise MixedCorrectionConfigError(
            "去料质量上限不得为负", {"remove_mass_limit": r_limit}
        )
    # 实际去料容量不超过各孔有效上限之和
    removable = min(r_limit, removable_capacity)

    # 默认单面改变量上限 = 实际加重容量 + 实际去料容量（不对二者之和额外绑定）
    c_limit = (
        float(add_mass_limit) + removable
        if change_mass_limit is None else float(change_mass_limit)
    )
    if c_limit < -EPS:
        raise MixedCorrectionConfigError(
            "单面改变量上限不得为负", {"change_mass_limit": c_limit}
        )

    return PlaneSpec(
        add_hole_angles=add_angles,
        weight_specs=specs,
        drill_holes=holes,
        existing_weights=existing,
        add_mass_limit=float(max(add_mass_limit, 0.0)),
        remove_mass_limit=max(r_limit, 0.0),
        change_mass_limit=max(c_limit, 0.0),
        mass_per_mm=mass_per_mm,
        max_weights_per_plane=max_weights_per_plane,
    )


# ---------------------------------------------------------------- 单面枚举


def enumerate_add(
    spec: PlaneSpec, add_angles: list[float]
) -> list[PlaneSelection]:
    """在指定加重孔上枚举可安装组合（含空组合）。

    已有配重占据的孔位不可再加重；总加重质量 ≤ add_mass_limit。
    """
    selections = [PlaneSelection(actions=[], vector=0j, add_mass=0.0, remove_mass=0.0)]
    occupied = {round(normalize_angle(w.hole_angle), 6) for w in spec.existing_weights}
    free_angles = [a for a in add_angles if round(a, 6) not in occupied]
    if not free_angles or spec.add_mass_limit <= EPS:
        return selections

    for k in range(1, spec.max_weights_per_plane + 1):
        if k > len(free_angles):
            break
        if comb(len(free_angles), k) * len(spec.weight_specs) ** k > ENUMERATION_CAP:
            break
        for idx in combinations(range(len(free_angles)), k):
            angles = [free_angles[i] for i in idx]
            for masses in product(spec.weight_specs, repeat=k):
                if sum(masses) > spec.add_mass_limit + 1e-9:
                    continue
                actions = [MixedAction("add", a, float(m)) for a, m in zip(angles, masses)]
                selections.append(
                    PlaneSelection(
                        actions=actions,
                        vector=complex(sum(a.vector for a in actions)),
                        add_mass=float(sum(masses)),
                        remove_mass=0.0,
                    )
                )
    return selections


def enumerate_remove(spec: PlaneSpec, removal_step: float | None) -> list[PlaneSelection]:
    """枚举钻削去料组合（含空组合）：每孔至多一处去料，深度按 step 离散。"""
    selections = [PlaneSelection(actions=[], vector=0j, add_mass=0.0, remove_mass=0.0)]
    if not spec.drill_holes or spec.remove_mass_limit <= EPS:
        return selections
    if not removal_step or removal_step <= 0:
        raise MixedCorrectionConfigError(
            "去料离散步长缺失或非正", {"removal_step": removal_step}
        )

    # 每孔可选去料质量档；厚度/每孔上限在此过滤
    hole_options: list[list[MixedAction]] = []
    for h in spec.drill_holes:
        eff = spec.effective_hole_limit(h)
        n = int(np.floor(eff / removal_step + 1e-9))
        options = [
            MixedAction("remove", h.hole_angle, round(removal_step * k, 9))
            for k in range(1, n + 1)
        ]
        if options:
            hole_options.append(options)
    if not hole_options:
        return selections

    max_options = max(len(o) for o in hole_options)
    for k in range(1, min(len(hole_options), spec.max_weights_per_plane) + 1):
        if comb(len(hole_options), k) * max_options ** k > ENUMERATION_CAP:
            break
        for idx in combinations(range(len(hole_options)), k):
            for chosen in product(*(hole_options[i] for i in idx)):
                total = sum(a.mass for a in chosen)
                if total > spec.remove_mass_limit + 1e-9:
                    continue
                actions = list(chosen)
                selections.append(
                    PlaneSelection(
                        actions=actions,
                        vector=complex(sum(a.vector for a in actions)),
                        add_mass=0.0,
                        remove_mass=float(total),
                    )
                )
    return selections


def enumerate_plane(
    spec: PlaneSpec, removal_step: float, target: complex | None = None,
    keep: int = 120,
) -> list[PlaneSelection]:
    """单面全部候选：加重组合 × 去料组合，受单面改变量上限约束。

    为避免“加重组合数 × 去料组合数”爆炸，先分别按与连续目标的接近程度
    剪枝（空组合恒保留；去料目标取反向，因去料向量本身带负号），交叉后
    再按合成向量与目标的距离剪到 keep 个。
    """
    if removal_step is not None and removal_step <= 0:
        raise MixedCorrectionConfigError(
            "去料离散步长必须为正数", {"removal_step": removal_step}
        )
    adds = enumerate_add(spec, spec.add_hole_angles)
    removes = enumerate_remove(spec, removal_step)
    if target is not None:
        adds = _prune(adds, target, keep)
        # 去料候选的向量是 −m·e^{iθ}，最接近目标的去料方向对应 −target
        removes = _prune(removes, -target, keep)

    out: list[PlaneSelection] = []
    for a_sel, r_sel in product(adds, removes):
        change = a_sel.add_mass + r_sel.remove_mass
        if change > spec.change_mass_limit + 1e-9:
            continue
        out.append(
            PlaneSelection(
                actions=a_sel.actions + r_sel.actions,
                vector=complex(a_sel.vector + r_sel.vector),
                add_mass=a_sel.add_mass,
                remove_mass=r_sel.remove_mass,
            )
        )
    if target is not None:
        out = _prune(out, target, keep)
    return out


def _prune(selections: list[PlaneSelection], target: complex,
           keep: int) -> list[PlaneSelection]:
    """按与连续目标的接近程度剪枝；空组合始终保留（与 discrete 同策略）。"""
    empty = [s for s in selections if not s.actions]
    rest = sorted(
        (s for s in selections if s.actions),
        key=lambda s: (abs(s.vector - target), s.change_mass),
    )
    return empty + rest[:keep]


# ---------------------------------------------------------------- 可行性诊断


def _plane_blockers_from_spec(
    plane_name: str, spec: PlaneSpec, required_mass: float,
) -> list[dict]:
    """直接依据几何/限值判断该面是否无任何可行动作（不构造大规模枚举）。"""
    blockers: list[dict] = []
    occupied = [
        {"hole_angle": normalize_angle(w.hole_angle), "mass": w.mass}
        for w in spec.existing_weights
    ]
    existing_total = sum(w.mass for w in spec.existing_weights)

    free_add = [
        a for a in spec.add_hole_angles
        if round(a, 6) not in {round(w.hole_angle, 6) for w in spec.existing_weights}
    ]
    can_add = (
        bool(free_add)
        and spec.add_mass_limit > EPS
        and min(spec.weight_specs) <= spec.add_mass_limit + 1e-9
        and spec.change_mass_limit > EPS
    )
    if not can_add:
        if not spec.add_hole_angles:
            blockers.append({"code": "no_add_holes", "plane": plane_name})
        elif not free_add:
            blockers.append({
                "code": "all_add_holes_occupied",
                "plane": plane_name,
                "occupied_holes": occupied,
            })
        if spec.add_mass_limit <= EPS:
            blockers.append({
                "code": "add_mass_limit_saturated",
                "plane": plane_name,
                "add_mass_limit": spec.add_mass_limit,
                "existing_mass": existing_total,
                "occupied_holes": occupied,
            })
        elif free_add and min(spec.weight_specs) > spec.add_mass_limit + 1e-9:
            blockers.append({
                "code": "add_mass_limit_below_min_spec",
                "plane": plane_name,
                "add_mass_limit": spec.add_mass_limit,
                "min_weight_spec": min(spec.weight_specs),
            })

    usable = [h for h in spec.drill_holes if spec.effective_hole_limit(h) > EPS]
    can_remove = (
        bool(usable) and spec.remove_mass_limit > EPS
        and spec.change_mass_limit > EPS
    )
    if not can_remove:
        if not spec.drill_holes:
            blockers.append({"code": "no_drill_holes", "plane": plane_name})
        elif not usable:
            blockers.append({
                "code": "no_drillable_hole",
                "plane": plane_name,
                "holes": [
                    {
                        "hole_angle": h.hole_angle,
                        "current_thickness": h.current_thickness,
                        "min_remaining_thickness": h.min_remaining_thickness,
                        "removal_limit": h.removal_limit,
                        "effective_limit": spec.effective_hole_limit(h),
                    }
                    for h in spec.drill_holes
                ],
            })
        if spec.drill_holes and usable and spec.remove_mass_limit <= EPS:
            blockers.append({
                "code": "remove_mass_limit_zero",
                "plane": plane_name,
                "remove_mass_limit": spec.remove_mass_limit,
            })

    if spec.change_mass_limit <= EPS:
        blockers.append({
            "code": "change_mass_limit_zero",
            "plane": plane_name,
            "change_mass_limit": spec.change_mass_limit,
        })

    if not blockers and not (can_add or can_remove):  # pragma: no cover - 防御
        blockers.append({
            "code": "constraints_conflict",
            "plane": plane_name,
            "add_mass_limit": spec.add_mass_limit,
            "remove_mass_limit": spec.remove_mass_limit,
            "change_mass_limit": spec.change_mass_limit,
            "required_resultant_mass": required_mass,
        })
    return blockers


def _plane_blockers(
    plane_name: str, spec: PlaneSpec, selections: list[PlaneSelection],
    required_mass: float,
) -> list[dict]:
    """枚举后兜底诊断（正常路径使用 _plane_blockers_from_spec）。"""
    if any(s.actions for s in selections):
        return []
    return _plane_blockers_from_spec(plane_name, spec, required_mass)


# ---------------------------------------------------------------- 评估


def _ratio(headroom: float, capacity: float) -> float:
    """相对安全余量：剩余容量/容量；容量为 0 时按 0（无余量可言）。"""
    if capacity <= EPS:
        return 0.0
    return float(max(min(headroom / capacity, 1.0), -1.0))


def _plane_safety(
    plane_name: str, spec: PlaneSelection, pspec: PlaneSpec,
) -> tuple[dict, list[tuple[float, dict]]]:
    """计算单面安全余量，返回 (JSON, 带标签的相对余量列表)。"""
    holes_by_angle = {round(h.hole_angle, 6): h for h in pspec.drill_holes}
    ratios: list[tuple[float, dict]] = []

    add_head = pspec.add_mass_limit - spec.add_mass
    rem_head = pspec.remove_mass_limit - spec.remove_mass
    chg_head = pspec.change_mass_limit - spec.change_mass
    ratios.append((
        _ratio(add_head, pspec.add_mass_limit),
        {"plane": plane_name, "constraint": "add_mass_limit",
         "headroom": float(max(add_head, 0.0))},
    ))
    ratios.append((
        _ratio(rem_head, pspec.remove_mass_limit),
        {"plane": plane_name, "constraint": "remove_mass_limit",
         "headroom": float(max(rem_head, 0.0))},
    ))
    ratios.append((
        _ratio(chg_head, pspec.change_mass_limit),
        {"plane": plane_name, "constraint": "change_mass_limit",
         "headroom": float(max(chg_head, 0.0))},
    ))

    holes_out = []
    for a in spec.actions:
        if a.kind != "remove":
            continue
        h = holes_by_angle[round(a.hole_angle, 6)]
        depth = a.mass / pspec.mass_per_mm if pspec.mass_per_mm else None
        remaining = h.current_thickness - depth if depth is not None else None
        margin = (
            remaining - h.min_remaining_thickness if remaining is not None else None
        )
        capacity = h.current_thickness - h.min_remaining_thickness
        ratio = _ratio(margin, capacity) if margin is not None else 0.0
        ratios.append((
            ratio,
            {"plane": plane_name, "constraint": "min_remaining_thickness",
             "hole_angle": a.hole_angle, "headroom": float(max(margin, 0.0))},
        ))
        holes_out.append({
            "hole_angle": a.hole_angle,
            "removed_mass": a.mass,
            "drill_depth": depth,
            "remaining_thickness": remaining,
            "min_remaining_thickness": h.min_remaining_thickness,
            "thickness_margin": margin,
            "thickness_margin_ratio": ratio,
        })

    out = {
        "plane": plane_name,
        "add_mass": spec.add_mass,
        "add_mass_limit": pspec.add_mass_limit,
        "add_headroom": float(max(add_head, 0.0)),
        "remove_mass": spec.remove_mass,
        "remove_mass_limit": pspec.remove_mass_limit,
        "remove_headroom": float(max(rem_head, 0.0)),
        "change_mass": spec.change_mass,
        "change_mass_limit": pspec.change_mass_limit,
        "change_headroom": float(max(chg_head, 0.0)),
        "drill_holes": holes_out,
    }
    return out, ratios


def evaluate_mixed(
    *,
    alpha: np.ndarray,
    v0: np.ndarray,
    plane_selections: list[PlaneSelection],
    plane_names: list[str],
    plane_specs: list[PlaneSpec],
    sensor_names: list[str],
    amp_error: float,
    phase_error_deg: float,
    angle_tolerance_deg: float,
    baseline_amplitudes: np.ndarray,
    build_details: bool = True,
) -> MixedCandidate:
    """评估一个给定的双面选择（枚举候选与确认后的动作走同一条路径）。

    build_details=False 时只计算排序所需的标量（残振/改变量/最差值/安全
    余量），不构建逐动作明细，供大规模候选排序使用；最终返回前再对 top
    候选以 build_details=True 重算。
    """
    n_planes = len(plane_names)
    weights = np.array([s.vector for s in plane_selections], dtype=complex)
    pred = v0 + alpha @ weights
    changes = [s.change_mass for s in plane_selections]
    wc = worst_case_residual(
        pred, alpha, changes, baseline_amplitudes,
        amp_error, phase_error_deg, angle_tolerance_deg,
    )

    all_ratios: list[tuple[float, dict]] = []
    plane_safety: list[dict] = []
    for p in range(n_planes):
        safety, ratios = _plane_safety(
            plane_names[p], plane_selections[p], plane_specs[p]
        )
        plane_safety.append(safety)
        all_ratios.extend(ratios)
    min_ratio, critical = min(all_ratios, key=lambda t: t[0])

    evaluated: list[EvaluatedAction] = []
    if build_details:
        for p, sel in enumerate(plane_selections):
            holes_by = {round(h.hole_angle, 6): h for h in plane_specs[p].drill_holes}
            for a in sel.actions:
                effects = {
                    sensor_names[i]: complex(alpha[i, p] * a.vector)
                    for i in range(len(sensor_names))
                }
                ea = EvaluatedAction(
                    kind=a.kind, hole_angle=a.hole_angle, mass=a.mass,
                    vector=a.vector, plane=plane_names[p], effects=effects,
                )
                if a.kind == "remove":
                    h = holes_by[round(a.hole_angle, 6)]
                    ea.drill_depth = (
                        a.mass / plane_specs[p].mass_per_mm
                        if plane_specs[p].mass_per_mm else None
                    )
                    ea.remaining_thickness = h.current_thickness - ea.drill_depth
                    ea.thickness_margin = ea.remaining_thickness - h.min_remaining_thickness
                    capacity = h.current_thickness - h.min_remaining_thickness
                    ea.thickness_margin_ratio = _ratio(ea.thickness_margin, capacity)
                evaluated.append(ea)

    actions = [a for sel in plane_selections for a in sel.actions]
    return MixedCandidate(
        actions=actions,
        weights=weights,
        predicted_residual=pred,
        predicted_metric=float(np.max(np.abs(pred))),
        total_change=float(sum(changes)),
        worst_case=float(wc),
        min_safety_margin=float(min_ratio),
        critical_constraint=critical,
        plane_safety=plane_safety if build_details else [],
        evaluated_actions=evaluated,
        plane_selections=list(plane_selections),
    )


# ---------------------------------------------------------------- 联合搜索


def _plane_ratios(pspec: PlaneSpec, sel: PlaneSelection) -> list[tuple[float, dict]]:
    """单面全部相对安全余量（轻量版，供大规模排序）。"""
    ratios: list[tuple[float, dict]] = [
        (_ratio(pspec.add_mass_limit - sel.add_mass, pspec.add_mass_limit),
         {"constraint": "add_mass_limit"}),
        (_ratio(pspec.remove_mass_limit - sel.remove_mass, pspec.remove_mass_limit),
         {"constraint": "remove_mass_limit"}),
        (_ratio(pspec.change_mass_limit - sel.change_mass, pspec.change_mass_limit),
         {"constraint": "change_mass_limit"}),
    ]
    if pspec.mass_per_mm:
        holes_by = {round(h.hole_angle, 6): h for h in pspec.drill_holes}
        for a in sel.actions:
            if a.kind != "remove":
                continue
            h = holes_by[round(a.hole_angle, 6)]
            capacity = h.current_thickness - h.min_remaining_thickness
            margin = capacity - a.mass / pspec.mass_per_mm
            ratios.append((
                _ratio(margin, capacity),
                {"constraint": "min_remaining_thickness",
                 "hole_angle": a.hole_angle},
            ))
    return ratios


def search_mixed_corrections(
    *,
    alpha: np.ndarray,
    v0: np.ndarray,
    plane_specs: list[PlaneSpec],
    plane_names: list[str],
    sensor_names: list[str],
    weight_specs: list[float],
    removal_step: float,
    amp_error: float,
    phase_error_deg: float,
    angle_tolerance_deg: float,
    baseline_amplitudes: np.ndarray,
    max_weights_per_plane: int = 3,
    keep_per_plane: int = 120,
    top: int = 10,
) -> list[MixedCandidate]:
    """联合枚举双面加重/去料组合并排序。

    排序：(预测残振升序, 总改变量升序, 最小相对安全余量降序)。
    无解时抛 NoFeasibleMixedCorrection，details 指出冲突的校正面与约束。
    """
    n_planes = alpha.shape[1]

    # 连续理想校正量作为剪枝目标（无约束最小二乘；约束在枚举中精确处理）
    target, *_ = np.linalg.lstsq(alpha, -v0, rcond=None)

    # 可行性诊断直接依据几何/限值（不构造全枚举，避免组合爆炸）
    per_plane: list[list[PlaneSelection]] = []
    all_blockers: list[dict] = []
    for p in range(n_planes):
        blockers = _plane_blockers_from_spec(
            plane_names[p], plane_specs[p], abs(target[p])
        )
        can_act = not blockers
        if not can_act and abs(target[p]) > EPS:
            all_blockers.extend(blockers)
        per_plane.append(
            enumerate_plane(
                plane_specs[p], removal_step, target=target[p], keep=keep_per_plane
            )
        )

    if all_blockers:
        raise NoFeasibleMixedCorrection(
            "存在校正面在孔位/质量/厚度/单面改变量约束下无任何可行动作",
            {"planes": all_blockers,
             "required_resultant_mass": {
                 plane_names[p]: float(abs(target[p])) for p in range(n_planes)
             }},
        )

    # ---- 轻量排序：只算标量（残振/改变量/最差值/安全余量）
    alpha_abs = np.abs(alpha)
    k_angle = 2.0 * np.sin(np.deg2rad(angle_tolerance_deg) / 2.0)
    k_meas = amp_error + 2.0 * np.sin(np.deg2rad(phase_error_deg) / 2.0)
    base = np.asarray(baseline_amplitudes, dtype=float) * k_meas

    light: list[tuple] = []
    for combo in product(*per_plane):
        w = np.array([s.vector for s in combo], dtype=complex)
        pred = v0 + alpha @ w
        changes = [s.change_mass for s in combo]
        metric = float(np.max(np.abs(pred)))
        wc = float(np.max(np.abs(pred) + alpha_abs @ (k_angle * np.array(changes)) + base))
        total_change = float(sum(changes))
        min_ratio = 1.0
        for p, sel in enumerate(combo):
            r, _ = min(_plane_ratios(plane_specs[p], sel), key=lambda t: t[0])
            if r < min_ratio:
                min_ratio = r
        light.append(
            (round(metric, 6), round(total_change, 6), round(-min_ratio, 6),
             metric, total_change, wc, min_ratio, combo)
        )

    light.sort(key=lambda t: t[:3])
    winners = light[:top]

    # ---- 仅对最终 top 候选构建完整明细（逐动作贡献/逐面安全 JSON）
    candidates: list[MixedCandidate] = []
    for _, _, _, _, _, _, _, combo in winners:
        candidates.append(
            evaluate_mixed(
                alpha=alpha,
                v0=v0,
                plane_selections=list(combo),
                plane_names=plane_names,
                plane_specs=plane_specs,
                sensor_names=sensor_names,
                amp_error=amp_error,
                phase_error_deg=phase_error_deg,
                angle_tolerance_deg=angle_tolerance_deg,
                baseline_amplitudes=baseline_amplitudes,
            )
        )
    return candidates


def assemble_plane_actions(
    spec: PlaneSpec,
    actions: list[MixedAction],
    removal_step: float | None,
    plane_name: str,
) -> PlaneSelection:
    """复核单面动作：孔位、质量规格、每孔去料上限/厚度、单面各限值。"""
    add_holes = {round(a, 6): a for a in spec.add_hole_angles}
    drill = {round(h.hole_angle, 6): h for h in spec.drill_holes}
    specs_set = {round(m, 9) for m in spec.weight_specs}
    occupied = {round(w.hole_angle, 6) for w in spec.existing_weights}

    add_mass = 0.0
    remove_mass = 0.0
    seen_add: set[float] = set()
    seen_remove: set[float] = set()
    vector = 0j
    out: list[MixedAction] = []

    for a in actions:
        ang = round(a.hole_angle, 6)
        if a.kind == "add":
            if ang not in add_holes:
                raise MixedCorrectionConfigError(
                    "加重孔不在可用加重孔位中",
                    {"plane": plane_name, "hole_angle": a.hole_angle,
                     "available_add_holes": spec.add_hole_angles},
                )
            if ang in occupied:
                raise MixedCorrectionConfigError(
                    "加重孔已被已有配重占据",
                    {"plane": plane_name, "hole_angle": a.hole_angle},
                )
            if ang in seen_add:
                raise MixedCorrectionConfigError(
                    "同一加重孔出现两个加重动作",
                    {"plane": plane_name, "hole_angle": a.hole_angle},
                )
            if not any(abs(a.mass - m) <= 1e-6 for m in specs_set):
                raise MixedCorrectionConfigError(
                    "加重质量不在配重规格表中",
                    {"plane": plane_name, "mass": a.mass,
                     "weight_specs": spec.weight_specs},
                )
            seen_add.add(ang)
            add_mass += a.mass
        elif a.kind == "remove":
            if ang not in drill:
                raise MixedCorrectionConfigError(
                    "去料孔不在可钻削孔位中",
                    {"plane": plane_name, "hole_angle": a.hole_angle},
                )
            if ang in seen_remove:
                raise MixedCorrectionConfigError(
                    "同一孔位出现两个去料动作",
                    {"plane": plane_name, "hole_angle": a.hole_angle},
                )
            h = drill[ang]
            eff = spec.effective_hole_limit(h)
            if a.mass > eff + 1e-7:
                raise MixedCorrectionConfigError(
                    "去料质量超过该孔有效去料上限（每孔上限/最小剩余厚度）",
                    {"plane": plane_name, "hole_angle": a.hole_angle,
                     "mass": a.mass, "effective_limit": eff,
                     "removal_limit": h.removal_limit,
                     "min_remaining_thickness": h.min_remaining_thickness},
                )
            if removal_step:
                n_steps = a.mass / removal_step
                if abs(n_steps - round(n_steps)) > 1e-6 * max(1.0, abs(n_steps)):
                    raise MixedCorrectionConfigError(
                        "去料质量不是离散步长的整数倍",
                        {"plane": plane_name, "hole_angle": a.hole_angle,
                         "mass": a.mass, "removal_step": removal_step},
                    )
            seen_remove.add(ang)
            remove_mass += a.mass
        else:  # pragma: no cover - schema 已限制
            raise MixedCorrectionConfigError(
                "未知动作类型", {"plane": plane_name, "kind": a.kind}
            )
        out.append(a)
        vector += a.vector

    if add_mass > spec.add_mass_limit + 1e-9:
        raise MixedCorrectionConfigError(
            "加重总质量超过该面上限",
            {"plane": plane_name, "add_mass": add_mass,
             "add_mass_limit": spec.add_mass_limit},
        )
    if remove_mass > spec.remove_mass_limit + 1e-9:
        raise MixedCorrectionConfigError(
            "去料总质量超过该面上限",
            {"plane": plane_name, "remove_mass": remove_mass,
             "remove_mass_limit": spec.remove_mass_limit},
        )
    if add_mass + remove_mass > spec.change_mass_limit + 1e-9:
        raise MixedCorrectionConfigError(
            "单面总改变量超过上限",
            {"plane": plane_name, "change_mass": add_mass + remove_mass,
             "change_mass_limit": spec.change_mass_limit},
        )

    return PlaneSelection(
        actions=out, vector=complex(vector),
        add_mass=add_mass, remove_mass=remove_mass,
    )
