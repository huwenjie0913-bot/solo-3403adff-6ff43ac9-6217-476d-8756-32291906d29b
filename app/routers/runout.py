"""慢转轴跳档案路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import services
from ..database import get_db
from ..schemas import RunoutProfileCreate, RunoutProfileOut, RunoutRecordCreate

router = APIRouter(prefix="/api/batches", tags=["runout"])


@router.post("/{batch_id}/runout-profiles",
             response_model=RunoutProfileOut, status_code=201)
def create_runout_profile(
    batch_id: int, payload: RunoutProfileCreate, db: Session = Depends(get_db)
):
    """为批次建立慢转轴跳档案（可随档案一次录入慢转记录）。

    档案保存逐测点复矢量均值、离散度与有效性问题；存在问题时仍可建档，
    但标定/求解/复测指定该档案会被拒绝。
    """
    return services.create_runout_profile(db, batch_id, payload)


@router.get("/{batch_id}/runout-profiles",
            response_model=list[RunoutProfileOut])
def list_runout_profiles(batch_id: int, db: Session = Depends(get_db)):
    batch = services.get_batch_or_404(db, batch_id)
    return batch.runout_profiles


@router.get("/{batch_id}/runout-profiles/{profile_id}",
            response_model=RunoutProfileOut)
def get_runout_profile(batch_id: int, profile_id: int, db: Session = Depends(get_db)):
    batch = services.get_batch_or_404(db, batch_id)
    return services._get_profile_or_404(db, batch, profile_id)


@router.post("/{batch_id}/runout-profiles/{profile_id}/records",
             response_model=RunoutProfileOut, status_code=201)
def add_runout_record(
    batch_id: int,
    profile_id: int,
    payload: RunoutRecordCreate,
    db: Session = Depends(get_db),
):
    """向轴跳档案追加一条慢转记录，并重算均值、离散度与有效性问题。"""
    return services.add_runout_record(db, batch_id, profile_id, payload)
