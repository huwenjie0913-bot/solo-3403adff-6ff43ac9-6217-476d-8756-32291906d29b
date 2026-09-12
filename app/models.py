"""ORM 模型：批次、运行、影响系数标定、方案、复测。"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Batch(Base):
    """平衡批次：一台设备的一次双面平衡任务及其全部配置约束。"""

    __tablename__ = "batches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    reference_speed: Mapped[float] = mapped_column(Float)          # 基准转速 rpm
    speed_tolerance: Mapped[float] = mapped_column(Float)          # 允许转速偏差 rpm
    planes: Mapped[list] = mapped_column(JSON)                     # [{name, correction_radius, hole_angles, mass_limit}]
    sensors: Mapped[list] = mapped_column(JSON)                    # [{name}]
    weight_specs: Mapped[list] = mapped_column(JSON)               # 可用配重质量规格 [g]
    amp_error: Mapped[float] = mapped_column(Float)                # 幅值相对误差
    phase_error: Mapped[float] = mapped_column(Float)              # 相位误差 度
    angle_tolerance: Mapped[float] = mapped_column(Float)          # 安装角公差 度
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    runs: Mapped[list["Run"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", order_by="Run.id"
    )
    runout_profiles: Mapped[list["RunoutProfile"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan",
        order_by="RunoutProfile.id",
    )
    calibrations: Mapped[list["Calibration"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan",
        order_by="Calibration.id",
    )
    solutions: Mapped[list["Solution"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", order_by="Solution.id"
    )
    mixed_plans: Mapped[list["MixedPlan"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", order_by="MixedPlan.id"
    )

    @property
    def calibration(self) -> "Calibration | None":
        """最新一次标定（不同轴跳档案可产生多份标定）。"""
        return self.calibrations[-1] if self.calibrations else None


class Run(Base):
    """一次运行：基线或试重。测量与试重以 JSON 存储，保留来源信息。"""

    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(20))                  # baseline | trial
    speed: Mapped[float] = mapped_column(Float)                    # 实测转速 rpm
    phase_reference: Mapped[str] = mapped_column(String(50))       # 相位基准约定
    measurements: Mapped[list] = mapped_column(JSON)               # [{sensor, amplitude, phase}]
    trial_weights: Mapped[list] = mapped_column(JSON, default=list)  # [{plane, mass, angle}]
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    batch: Mapped[Batch] = relationship(back_populates="runs")


class RunoutProfile(Base):
    """慢转轴跳档案：一组慢转记录的逐测点复矢量统计与有效性问题。"""

    __tablename__ = "runout_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(200))
    slow_roll_speed_limit: Mapped[float] = mapped_column(Float)   # 慢转转速上限 rpm
    dispersion_limit: Mapped[float] = mapped_column(Float)        # 重复测量离散度阈值（振幅单位）
    summary: Mapped[dict] = mapped_column(JSON)                   # 逐测点均值/离散度统计
    issues: Mapped[list] = mapped_column(JSON, default=list)      # 档案有效性问题清单
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    batch: Mapped[Batch] = relationship(back_populates="runout_profiles")
    records: Mapped[list["RunoutRecord"]] = relationship(
        back_populates="profile", cascade="all, delete-orphan",
        order_by="RunoutRecord.id",
    )

    @property
    def usable(self) -> bool:
        """档案是否通过全部使用前校验（无任何有效性问题）。"""
        return not self.issues


class RunoutRecord(Base):
    """一条慢转记录：转速、相位基准及各测点 1X 振幅/相位。"""

    __tablename__ = "runout_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    profile_id: Mapped[int] = mapped_column(
        ForeignKey("runout_profiles.id", ondelete="CASCADE")
    )
    speed: Mapped[float] = mapped_column(Float)
    phase_reference: Mapped[str] = mapped_column(String(50))
    measurements: Mapped[list] = mapped_column(JSON)   # [{sensor, amplitude, phase}]
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    profile: Mapped[RunoutProfile] = relationship(back_populates="records")


class Calibration(Base):
    """影响系数标定结果及其测量来源。"""

    __tablename__ = "calibrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"))
    runout_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("runout_profiles.id", ondelete="SET NULL"), nullable=True
    )
    coefficients: Mapped[dict] = mapped_column(JSON)   # {sensor: {plane: {real, imag, magnitude, phase}}}
    residuals: Mapped[list] = mapped_column(JSON)      # [{run_id, sensor, amplitude, phase, relative}]
    condition: Mapped[float] = mapped_column(Float)
    provenance: Mapped[dict] = mapped_column(JSON)     # 基线/试重运行 id、测点、校正面、时间
    # 轴跳补偿快照：使用的档案 id/名称、限值、逐测点原始/补偿/净振动与可分辨范围
    runout_compensation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    batch: Mapped[Batch] = relationship(back_populates="calibrations")


class Solution(Base):
    """校正方案：连续解或离散可安装组合。"""

    __tablename__ = "solutions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"))
    runout_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("runout_profiles.id", ondelete="SET NULL"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(20))                  # continuous | discrete
    rank: Mapped[int] = mapped_column(Integer, default=0)
    weights: Mapped[list] = mapped_column(JSON)          # [{plane, mass, angle, assignments?}]
    predicted_residual: Mapped[dict] = mapped_column(JSON)  # {sensor: {amplitude, phase}}
    predicted_metric: Mapped[float] = mapped_column(Float)
    total_mass: Mapped[float] = mapped_column(Float)
    worst_case: Mapped[float] = mapped_column(Float)
    # 净残振可分辨范围与逐测点/总体不可判定标记
    resolution: Mapped[dict] = mapped_column(JSON, default=dict)
    runout_compensation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    batch: Mapped[Batch] = relationship(back_populates="solutions")
    verifications: Mapped[list["Verification"]] = relationship(
        back_populates="solution", cascade="all, delete-orphan", order_by="Verification.id"
    )


class Verification(Base):
    """复测记录：安装方案后的实测振动与预测偏差对比。"""

    __tablename__ = "verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    solution_id: Mapped[int] = mapped_column(ForeignKey("solutions.id", ondelete="CASCADE"))
    runout_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("runout_profiles.id", ondelete="SET NULL"), nullable=True
    )
    speed: Mapped[float] = mapped_column(Float)
    measurements: Mapped[list] = mapped_column(JSON)   # [{sensor, amplitude, phase}]
    comparison: Mapped[dict] = mapped_column(JSON)     # 预测 vs 实测偏差（含轴跳补偿明细）
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    solution: Mapped[Solution] = relationship(back_populates="verifications")


class MixedPlan(Base):
    """加重/去料混合校正动作方案（确认后保存，带几何与约束快照）。"""

    __tablename__ = "mixed_plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"))
    calibration_id: Mapped[int] = mapped_column(
        ForeignKey("calibrations.id", ondelete="RESTRICT")
    )
    runout_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("runout_profiles.id", ondelete="SET NULL"), nullable=True
    )
    name: Mapped[str] = mapped_column(String(200), default="")
    # 确认后的动作：[{plane, kind: add|remove, hole_angle, mass, drill_depth?}]
    actions: Mapped[list] = mapped_column(JSON)
    predicted_residual: Mapped[dict] = mapped_column(JSON)   # {sensor: {amplitude, phase}}
    predicted_metric: Mapped[float] = mapped_column(Float)
    total_change: Mapped[float] = mapped_column(Float)       # 加重+去料总改变量 g
    worst_case: Mapped[float] = mapped_column(Float)
    min_safety_margin: Mapped[float] = mapped_column(Float)
    critical_constraint: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    plane_safety: Mapped[list] = mapped_column(JSON)         # 逐面安全余量
    action_effects: Mapped[list] = mapped_column(JSON)       # 逐动作对测点预测贡献
    resolution: Mapped[dict] = mapped_column(JSON, default=dict)
    # 几何快照：逐面加重孔/钻削孔/已有配重/各限值/换算/步长
    geometry_snapshot: Mapped[dict] = mapped_column(JSON)
    constraints_snapshot: Mapped[dict] = mapped_column(JSON)  # 公差与连续目标等
    runout_compensation: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    batch: Mapped[Batch] = relationship(back_populates="mixed_plans")
    verifications: Mapped[list["MixedVerification"]] = relationship(
        back_populates="plan", cascade="all, delete-orphan",
        order_by="MixedVerification.id",
    )


class MixedVerification(Base):
    """混合校正方案复测：逐项核对实际加重/去料并显示偏差。"""

    __tablename__ = "mixed_verifications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    plan_id: Mapped[int] = mapped_column(ForeignKey("mixed_plans.id", ondelete="CASCADE"))
    runout_profile_id: Mapped[int | None] = mapped_column(
        ForeignKey("runout_profiles.id", ondelete="SET NULL"), nullable=True
    )
    speed: Mapped[float] = mapped_column(Float)
    measurements: Mapped[list] = mapped_column(JSON)   # [{sensor, amplitude, phase}]
    actual_actions: Mapped[list] = mapped_column(JSON)  # 逐项实际动作及偏差
    reconciliation: Mapped[dict] = mapped_column(JSON)  # 动作核对与残振对比
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    plan: Mapped[MixedPlan] = relationship(back_populates="verifications")
