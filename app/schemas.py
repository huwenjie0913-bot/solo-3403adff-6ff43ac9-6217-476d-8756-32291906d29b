"""Pydantic 请求/响应模式。"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

# ---------------------------------------------------------------- 配置


class PlaneConfig(BaseModel):
    name: str = Field(..., examples=["P1"])
    correction_radius: float = Field(..., gt=0, description="校正半径 mm")
    hole_angles: list[float] = Field(..., description="可安装孔位角度（度）")
    mass_limit: float = Field(..., gt=0, description="该面总配重质量上限 g")

    @field_validator("hole_angles")
    @classmethod
    def _norm_angles(cls, v: list[float]) -> list[float]:
        angles = sorted({round(a % 360.0, 6) for a in v})
        if not angles:
            raise ValueError("每个校正面至少需要一个孔位角度")
        return angles


class SensorConfig(BaseModel):
    name: str = Field(..., examples=["DE"])


class BatchCreate(BaseModel):
    name: str
    description: str = ""
    reference_speed: float = Field(..., gt=0, description="基准转速 rpm")
    speed_tolerance: float = Field(30.0, ge=0, description="允许转速偏差 rpm")
    planes: list[PlaneConfig] = Field(..., min_length=2, max_length=2)
    sensors: list[SensorConfig] = Field(..., min_length=1)
    weight_specs: list[float] = Field(..., min_length=1, description="可用配重质量规格 g")
    amp_error: float = Field(0.02, ge=0, le=1, description="幅值相对误差")
    phase_error: float = Field(2.0, ge=0, le=45, description="相位误差 度")
    angle_tolerance: float = Field(5.0, ge=0, le=45, description="安装角公差 度")

    @field_validator("weight_specs")
    @classmethod
    def _positive_specs(cls, v: list[float]) -> list[float]:
        if any(m <= 0 for m in v):
            raise ValueError("配重规格必须为正数")
        return sorted(set(v))


class BatchOut(ORMModel):
    id: int
    name: str
    description: str
    reference_speed: float
    speed_tolerance: float
    planes: list[PlaneConfig]
    sensors: list[SensorConfig]
    weight_specs: list[float]
    amp_error: float
    phase_error: float
    angle_tolerance: float
    created_at: datetime
    run_count: int = 0
    calibrated: bool = False
    solution_count: int = 0
    runout_profile_count: int = 0


# ---------------------------------------------------------------- 运行


class MeasurementIn(BaseModel):
    sensor: str
    amplitude: float = Field(..., ge=0)
    phase: float = Field(..., description="相位 度")


class TrialWeightIn(BaseModel):
    plane: str
    mass: float = Field(..., gt=0)
    angle: float = Field(..., description="安装角度 度")


class RunCreate(BaseModel):
    kind: Literal["baseline", "trial"]
    speed: float = Field(..., gt=0, description="实测转速 rpm")
    phase_reference: str = Field("lag", description="相位基准约定，如 lag/lead")
    measurements: list[MeasurementIn] = Field(..., min_length=1)
    trial_weights: list[TrialWeightIn] = Field(default_factory=list)
    note: str = ""

    @model_validator(mode="after")
    def _check_kind(self):
        if self.kind == "baseline" and self.trial_weights:
            raise ValueError("基线运行不应包含试重")
        if self.kind == "trial" and not self.trial_weights:
            raise ValueError("试重运行必须包含至少一个试重")
        return self


class RunOut(ORMModel):
    id: int
    batch_id: int
    kind: str
    speed: float
    phase_reference: str
    measurements: list[MeasurementIn]
    trial_weights: list[TrialWeightIn]
    note: str
    created_at: datetime


# ---------------------------------------------------------------- 慢转轴跳


class RunoutRecordCreate(BaseModel):
    speed: float = Field(..., gt=0, description="慢转实测转速 rpm")
    phase_reference: str = Field("lag", description="相位基准约定，如 lag/lead")
    measurements: list[MeasurementIn] = Field(..., min_length=1)
    note: str = ""


class RunoutProfileCreate(BaseModel):
    name: str = Field(..., min_length=1)
    slow_roll_speed_limit: float = Field(
        ..., gt=0, description="慢转转速上限 rpm，超过即拒绝使用该档案"
    )
    dispersion_limit: float = Field(
        ..., ge=0, description="重复测量离散度阈值（与振幅同单位）"
    )
    records: list[RunoutRecordCreate] = Field(default_factory=list)
    note: str = ""


class RunoutRecordOut(ORMModel):
    id: int
    profile_id: int
    speed: float
    phase_reference: str
    measurements: list[MeasurementIn]
    note: str
    created_at: datetime


class RunoutProfileOut(ORMModel):
    id: int
    batch_id: int
    name: str
    slow_roll_speed_limit: float
    dispersion_limit: float
    summary: dict
    issues: list[dict]
    usable: bool
    note: str
    created_at: datetime
    records: list[RunoutRecordOut] = Field(default_factory=list)


# ---------------------------------------------------------------- 标定


class ComplexValue(BaseModel):
    real: float
    imag: float
    magnitude: float
    phase: float


class CalibrationOut(ORMModel):
    batch_id: int
    runout_profile_id: int | None = None
    coefficients: dict[str, dict[str, ComplexValue]]  # sensor -> plane -> 系数
    residuals: list[dict]                              # 逐运行逐测点拟合残差
    condition: float
    provenance: dict                                   # 测量来源
    runout_compensation: dict | None = None            # 轴跳扣除快照（含逐测点原始/补偿/净振动）
    created_at: datetime


class CalibrateRequest(BaseModel):
    runout_profile_id: int | None = Field(
        None, description="标定前从各运行原始振动中扣除的轴跳档案 id"
    )


# ---------------------------------------------------------------- 方案


class WeightOut(BaseModel):
    plane: str
    mass: float
    angle: float
    assignments: list[dict] = Field(default_factory=list)


class SolutionOut(ORMModel):
    id: int
    batch_id: int
    runout_profile_id: int | None = None
    kind: str
    rank: int
    weights: list[WeightOut]
    predicted_residual: dict[str, dict[str, float]]  # sensor -> {amplitude, phase}
    predicted_metric: float
    total_mass: float
    worst_case: float
    resolution: dict = Field(default_factory=dict)   # 逐测点可分辨范围与不可判定标记
    runout_compensation: dict | None = None          # 基线轴跳扣除快照
    created_at: datetime


class SolveRequest(BaseModel):
    max_weights_per_plane: int = Field(3, ge=1, le=6)
    keep_per_plane: int = Field(120, ge=10, le=2000)
    top: int = Field(10, ge=1, le=100)
    runout_profile_id: int | None = Field(
        None, description="求解前先扣除的慢转轴跳档案 id；切换不改写历史标定/方案"
    )


class SolveResponse(BaseModel):
    continuous: SolutionOut
    discrete: list[SolutionOut]


# ---------------------------------------------------------------- 复测


class VerificationCreate(BaseModel):
    speed: float = Field(..., gt=0)
    measurements: list[MeasurementIn] = Field(..., min_length=1)
    phase_reference: str | None = Field(
        None, description="复测相位基准；须与轴跳档案/方案一致，缺省取批次基线约定"
    )
    runout_profile_id: int | None = Field(
        None, description="复测扣除的轴跳档案 id；缺省沿用方案所用档案"
    )
    note: str = ""


class VerificationOut(ORMModel):
    id: int
    solution_id: int
    runout_profile_id: int | None = None
    speed: float
    measurements: list[MeasurementIn]
    comparison: dict
    note: str
    created_at: datetime


# ---------------------------------------------------------------- 加重/去料混合校正


class ExistingWeightIn(BaseModel):
    hole_angle: float = Field(..., description="已有配重所在孔位角度 度")
    mass: float = Field(..., ge=0, description="已有配重质量 g")


class DrillHoleIn(BaseModel):
    hole_angle: float = Field(..., description="可钻削孔位角度 度")
    current_thickness: float = Field(..., ge=0, description="当前剩余厚度 mm")
    removal_limit: float = Field(..., gt=0, description="每孔去料质量上限 g")
    min_remaining_thickness: float = Field(
        0.0, ge=0, description="钻后最小允许剩余厚度 mm"
    )


class MixedPlaneIn(BaseModel):
    """逐面录入的可用加重孔、可钻削孔与已有配重等几何/约束。"""

    plane: str = Field(..., description="校正面名（须与批次配置一致）")
    add_hole_angles: list[float] = Field(
        default_factory=list, description="可用加重孔位角度 度"
    )
    drill_holes: list[DrillHoleIn] = Field(default_factory=list)
    existing_weights: list[ExistingWeightIn] = Field(default_factory=list)
    mass_per_mm: float | None = Field(
        None, gt=0, description="钻削去料质量/深度换算 g/mm（有钻削孔时必填）"
    )
    add_mass_limit: float | None = Field(
        None, ge=0, description="本次加重总质量上限 g；缺省=批次面质量上限−已有配重"
    )
    remove_mass_limit: float | None = Field(
        None, ge=0, description="本次去料总质量上限 g；缺省=各孔有效上限之和"
    )
    change_mass_limit: float | None = Field(
        None, ge=0, description="单面总改变量(加重+去料)上限 g；缺省不额外绑定"
    )

    @field_validator("add_hole_angles")
    @classmethod
    def _norm_add_angles(cls, v: list[float]) -> list[float]:
        return sorted({round(a % 360.0, 6) for a in v})


class MixedActionIn(BaseModel):
    plane: str
    kind: Literal["add", "remove"]
    hole_angle: float
    mass: float = Field(..., gt=0, description="加重质量或去料质量 g")


class MixedSearchRequest(BaseModel):
    planes: list[MixedPlaneIn] = Field(..., min_length=2, max_length=2)
    removal_step: float | None = Field(
        None, gt=0,
        description="去料质量离散步长 g；缺省取批次最小配重规格",
    )
    calibration_id: int | None = Field(
        None, description="沿用的标定 id；缺省取与所选轴跳档案匹配的标定，不自动新建"
    )
    runout_profile_id: int | None = Field(None)
    max_weights_per_plane: int = Field(3, ge=1, le=6)
    keep_per_plane: int = Field(120, ge=10, le=2000)
    top: int = Field(10, ge=1, le=100)


class MixedPlanCreate(MixedSearchRequest):
    actions: list[MixedActionIn] = Field(
        ..., description="确认保存的动作方案（逐项复核几何与约束快照）"
    )
    candidate_rank: int | None = Field(
        None, ge=1, description="搜索结果中的候选名次，仅作来源记录"
    )
    name: str = ""


class ActualMixedActionIn(BaseModel):
    plane: str
    kind: Literal["add", "remove"]
    hole_angle: float
    mass: float = Field(..., ge=0, description="实际加重/去料质量 g（未执行可为 0）")
    executed: bool = Field(True, description="该动作是否实际执行")


class MixedVerificationCreate(VerificationCreate):
    actual_actions: list[ActualMixedActionIn] = Field(
        ..., min_length=1, description="按方案逐项录入的实际加重/去料"
    )


class PlaneSafetyOut(BaseModel):
    plane: str
    add_mass: float
    add_mass_limit: float
    add_headroom: float
    remove_mass: float
    remove_mass_limit: float
    remove_headroom: float
    change_mass: float
    change_mass_limit: float
    change_headroom: float
    drill_holes: list[dict] = Field(default_factory=list)


class MixedActionOut(BaseModel):
    plane: str
    kind: str
    hole_angle: float
    mass: float
    drill_depth: float | None = None
    remaining_thickness: float | None = None
    thickness_margin: float | None = None


class MixedCandidateOut(BaseModel):
    rank: int
    actions: list[MixedActionOut]
    plane_weights: list[dict]
    predicted_residual: dict[str, dict[str, float]]
    predicted_metric: float
    total_change: float
    worst_case: float
    min_safety_margin: float
    critical_constraint: dict | None = None
    plane_safety: list[PlaneSafetyOut]
    action_effects: list[dict]
    resolution: dict = Field(default_factory=dict)


class MixedSearchResponse(BaseModel):
    calibration_id: int
    runout_profile_id: int | None = None
    phase_reference: str | None = None
    removal_step: float
    geometry_snapshot: dict
    candidates: list[MixedCandidateOut]


class MixedPlanOut(ORMModel):
    id: int
    batch_id: int
    calibration_id: int
    runout_profile_id: int | None = None
    name: str
    actions: list[dict]
    predicted_residual: dict[str, dict[str, float]]
    predicted_metric: float
    total_change: float
    worst_case: float
    min_safety_margin: float
    critical_constraint: dict | None = None
    plane_safety: list[dict]
    action_effects: list[dict]
    resolution: dict = Field(default_factory=dict)
    geometry_snapshot: dict
    constraints_snapshot: dict
    runout_compensation: dict | None = None
    note: str
    created_at: datetime
    verification_count: int = 0


class MixedVerificationOut(ORMModel):
    id: int
    plan_id: int
    runout_profile_id: int | None = None
    speed: float
    measurements: list[MeasurementIn]
    actual_actions: list[dict]
    reconciliation: dict
    note: str
    created_at: datetime
