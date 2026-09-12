"""FastAPI 应用入口。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .database import init_db
from .errors import DomainError
from .routers import batches, mixed, runout, solutions


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="双面现场动平衡 API",
    description=(
        "面向旋转设备检修的双面现场动平衡：慢转轴跳档案与矢量扣除，"
        "幅相差分拟合影响系数矩阵，连续配重求解与离散可安装组合搜索，"
        "加重/去料混合校正（去料以反向复矢量联合求解），"
        "公差最差情形评估与净振动可分辨判定，复测对比与 JSON 平衡记录导出。"
    ),
    version="1.2.0",
    lifespan=lifespan,
)


@app.exception_handler(DomainError)
async def domain_error_handler(_: Request, exc: DomainError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.code, "message": exc.message, "details": exc.details},
    )


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(batches.router)
app.include_router(runout.router)
app.include_router(solutions.router)
app.include_router(mixed.router)
