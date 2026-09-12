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
    calibration: Mapped["Calibration | None"] = relationship(
        back_populates="batch", cascade="all, delete-orphan", uselist=False
    )
    solutions: Mapped[list["Solution"]] = relationship(
        back_populates="batch", cascade="all, delete-orphan", order_by="Solution.id"
    )


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


class Calibration(Base):
    """影响系数标定结果及其测量来源。"""

    __tablename__ = "calibrations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"), unique=True)
    coefficients: Mapped[dict] = mapped_column(JSON)   # {sensor: {plane: {real, imag, magnitude, phase}}}
    residuals: Mapped[list] = mapped_column(JSON)      # [{run_id, sensor, amplitude, phase, relative}]
    condition: Mapped[float] = mapped_column(Float)
    provenance: Mapped[dict] = mapped_column(JSON)     # 基线/试重运行 id、测点、校正面、时间
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    batch: Mapped[Batch] = relationship(back_populates="calibration")


class Solution(Base):
    """校正方案：连续解或离散可安装组合。"""

    __tablename__ = "solutions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    batch_id: Mapped[int] = mapped_column(ForeignKey("batches.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(20))                  # continuous | discrete
    rank: Mapped[int] = mapped_column(Integer, default=0)
    weights: Mapped[list] = mapped_column(JSON)          # [{plane, mass, angle, assignments?}]
    predicted_residual: Mapped[dict] = mapped_column(JSON)  # {sensor: {amplitude, phase}}
    predicted_metric: Mapped[float] = mapped_column(Float)
    total_mass: Mapped[float] = mapped_column(Float)
    worst_case: Mapped[float] = mapped_column(Float)
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
    speed: Mapped[float] = mapped_column(Float)
    measurements: Mapped[list] = mapped_column(JSON)   # [{sensor, amplitude, phase}]
    comparison: Mapped[dict] = mapped_column(JSON)     # 预测 vs 实测偏差
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    solution: Mapped[Solution] = relationship(back_populates="verifications")
