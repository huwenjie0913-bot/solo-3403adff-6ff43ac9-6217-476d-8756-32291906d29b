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


# ---------------------------------------------------------------- 标定


class ComplexValue(BaseModel):
    real: float
    imag: float
    magnitude: float
    phase: float


class CalibrationOut(ORMModel):
    batch_id: int
    coefficients: dict[str, dict[str, ComplexValue]]  # sensor -> plane -> 系数
    residuals: list[dict]                              # 逐运行逐测点拟合残差
    condition: float
    provenance: dict                                   # 测量来源
    created_at: datetime


# ---------------------------------------------------------------- 方案


class WeightOut(BaseModel):
    plane: str
    mass: float
    angle: float
    assignments: list[dict] = Field(default_factory=list)


class SolutionOut(ORMModel):
    id: int
    batch_id: int
    kind: str
    rank: int
    weights: list[WeightOut]
    predicted_residual: dict[str, dict[str, float]]  # sensor -> {amplitude, phase}
    predicted_metric: float
    total_mass: float
    worst_case: float
    created_at: datetime


class SolveRequest(BaseModel):
    max_weights_per_plane: int = Field(3, ge=1, le=6)
    keep_per_plane: int = Field(120, ge=10, le=2000)
    top: int = Field(10, ge=1, le=100)


class SolveResponse(BaseModel):
    continuous: SolutionOut
    discrete: list[SolutionOut]


# ---------------------------------------------------------------- 复测


class VerificationCreate(BaseModel):
    speed: float = Field(..., gt=0)
    measurements: list[MeasurementIn] = Field(..., min_length=1)
    note: str = ""


class VerificationOut(ORMModel):
    id: int
    solution_id: int
    speed: float
    measurements: list[MeasurementIn]
    comparison: dict
    note: str
    created_at: datetime
