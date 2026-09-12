"""加重/去料混合校正路由：候选搜索、方案确认与复测逐项核对。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import services
from ..database import get_db
from ..errors import NotFoundError
from ..models import MixedPlan
from ..schemas import (
    MixedPlanCreate,
    MixedPlanOut,
    MixedSearchRequest,
    MixedSearchResponse,
    MixedVerificationCreate,
    MixedVerificationOut,
)

router = APIRouter(prefix="/api", tags=["mixed-correction"])


@router.post(
    "/batches/{batch_id}/mixed-corrections/search",
    response_model=MixedSearchResponse,
)
def search_mixed(batch_id: int, payload: MixedSearchRequest, db: Session = Depends(get_db)):
    """联合枚举加重与去料（反向复矢量）候选，返回预测残振/最差值/安全余量排序。

    不保存任何方案，也不改动既有运行、标定或加重方案。
    """
    response, *_ = services.search_mixed(db, batch_id, payload)
    return response


@router.post(
    "/batches/{batch_id}/mixed-corrections",
    response_model=MixedPlanOut,
    status_code=201,
)
def create_mixed_plan(
    batch_id: int, payload: MixedPlanCreate, db: Session = Depends(get_db)
):
    """确认并保存带几何与约束快照的混合校正动作方案。"""
    plan = services.create_mixed_plan(db, batch_id, payload)
    return services.plan_out(plan)


@router.get(
    "/batches/{batch_id}/mixed-corrections",
    response_model=list[MixedPlanOut],
)
def list_mixed_plans(batch_id: int, db: Session = Depends(get_db)):
    batch = services.get_batch_or_404(db, batch_id)
    return [services.plan_out(p) for p in batch.mixed_plans]


@router.get("/mixed-corrections/{plan_id}", response_model=MixedPlanOut)
def get_mixed_plan(plan_id: int, db: Session = Depends(get_db)):
    plan = db.get(MixedPlan, plan_id)
    if plan is None:
        raise NotFoundError(
            f"混合校正方案 {plan_id} 不存在", {"plan_id": plan_id}
        )
    return services.plan_out(plan)


@router.post(
    "/mixed-corrections/{plan_id}/verifications",
    response_model=MixedVerificationOut,
    status_code=201,
)
def add_mixed_verification(
    plan_id: int, payload: MixedVerificationCreate, db: Session = Depends(get_db)
):
    """录入复测：逐项核对实际加重/去料并显示偏差，再对比测点残振。"""
    return services.add_mixed_verification(db, plan_id, payload)
