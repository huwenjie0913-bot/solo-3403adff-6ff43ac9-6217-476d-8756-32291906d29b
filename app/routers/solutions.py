"""方案求解与复测路由。"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from .. import services
from ..database import get_db
from ..schemas import (
    SolutionOut,
    SolveRequest,
    SolveResponse,
    VerificationCreate,
    VerificationOut,
)

router = APIRouter(prefix="/api", tags=["solutions"])


@router.post("/batches/{batch_id}/solutions", response_model=SolveResponse, status_code=201)
def solve(batch_id: int, payload: SolveRequest | None = None, db: Session = Depends(get_db)):
    """计算连续配重并搜索可安装离散组合，按预测残振/总配重/最差结果排序。"""
    req = payload or SolveRequest()
    continuous, discrete = services.solve(db, batch_id, req)
    return SolveResponse(continuous=continuous, discrete=discrete)


@router.get("/batches/{batch_id}/solutions", response_model=list[SolutionOut])
def list_solutions(batch_id: int, db: Session = Depends(get_db)):
    batch = services.get_batch_or_404(db, batch_id)
    return sorted(batch.solutions, key=lambda s: (s.kind != "continuous", s.rank))


@router.post("/solutions/{solution_id}/verifications", response_model=VerificationOut, status_code=201)
def add_verification(solution_id: int, payload: VerificationCreate, db: Session = Depends(get_db)):
    """录入复测结果，返回预测与实测偏差对比。"""
    return services.add_verification(db, solution_id, payload)


@router.get("/solutions/{solution_id}", response_model=SolutionOut)
def get_solution(solution_id: int, db: Session = Depends(get_db)):
    from ..errors import NotFoundError
    from ..models import Solution

    sol = db.get(Solution, solution_id)
    if sol is None:
        raise NotFoundError(f"方案 {solution_id} 不存在", {"solution_id": solution_id})
    return sol
