"""批次、运行、标定与导出路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import services
from ..database import get_db
from ..models import Batch
from ..schemas import (
    BatchCreate,
    BatchOut,
    CalibrationOut,
    RunCreate,
    RunOut,
)

router = APIRouter(prefix="/api/batches", tags=["batches"])


def _batch_out(batch: Batch) -> BatchOut:
    return BatchOut(
        id=batch.id,
        name=batch.name,
        description=batch.description,
        reference_speed=batch.reference_speed,
        speed_tolerance=batch.speed_tolerance,
        planes=batch.planes,
        sensors=batch.sensors,
        weight_specs=batch.weight_specs,
        amp_error=batch.amp_error,
        phase_error=batch.phase_error,
        angle_tolerance=batch.angle_tolerance,
        created_at=batch.created_at,
        run_count=len(batch.runs),
        calibrated=batch.calibration is not None,
        solution_count=len(batch.solutions),
    )


@router.post("", response_model=BatchOut, status_code=201)
def create_batch(payload: BatchCreate, db: Session = Depends(get_db)):
    """创建平衡批次（含校正面、测点、配重规格与各项公差约束）。"""
    return _batch_out(services.create_batch(db, payload))


@router.get("", response_model=list[BatchOut])
def list_batches(db: Session = Depends(get_db)):
    return [_batch_out(b) for b in db.query(Batch).order_by(Batch.id).all()]


@router.get("/{batch_id}", response_model=BatchOut)
def get_batch(batch_id: int, db: Session = Depends(get_db)):
    return _batch_out(services.get_batch_or_404(db, batch_id))


@router.post("/{batch_id}/runs", response_model=RunOut, status_code=201)
def add_run(batch_id: int, payload: RunCreate, db: Session = Depends(get_db)):
    """录入基线或试重运行（幅相、转速、试重质量与角度）。"""
    return services.add_run(db, batch_id, payload)


@router.get("/{batch_id}/runs", response_model=list[RunOut])
def list_runs(batch_id: int, db: Session = Depends(get_db)):
    batch = services.get_batch_or_404(db, batch_id)
    return batch.runs


@router.post("/{batch_id}/calibrate", response_model=CalibrationOut)
def calibrate(batch_id: int, db: Session = Depends(get_db)):
    """按试重前后差分拟合影响系数矩阵，返回系数、拟合残差与测量来源。"""
    return services.calibrate(db, batch_id)


@router.get("/{batch_id}/calibration", response_model=CalibrationOut)
def get_calibration(batch_id: int, db: Session = Depends(get_db)):
    batch = services.get_batch_or_404(db, batch_id)
    if batch.calibration is None:
        from ..errors import CalibrationMissing

        raise CalibrationMissing("批次尚未标定", {"batch_id": batch_id})
    return batch.calibration


@router.get("/{batch_id}/export")
def export_batch(batch_id: int, db: Session = Depends(get_db)):
    """导出完整 JSON 平衡记录。"""
    return services.export_batch(db, batch_id)
