# 双面现场动平衡 API

面向旋转设备检修人员的双面现场动平衡 REST API。接收两个校正面、振动测点、基准转速以及基线/各次试重的幅相数据，按试重前后差分拟合影响系数矩阵，计算连续配重并搜索可安装的离散配重组合，支持复测对比与 JSON 平衡记录导出。

## 技术栈

Python 3.11 · FastAPI · Pydantic v2 · SQLAlchemy 2 · SQLite · NumPy · SciPy 数值栈（`numpy.linalg` 最小二乘）

## 启动

```bash
pip install -r requirements.txt
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- 交互文档：http://127.0.0.1:8000/docs
- 数据库默认 `./balancing.db`，可用环境变量 `BALANCING_DATABASE_URL` 覆盖
- 测试：`python3 -m pytest tests/ -q`

## 工作流程

1. **创建批次** `POST /api/batches` — 设备、基准转速、两个校正面（校正半径、孔位角度、单面质量上限）、测点、配重规格，以及允许转速偏差、幅值/相位误差、安装角公差。
2. **录入运行** `POST /api/batches/{id}/runs` — 一条基线 + 各次试重（振幅、相位、试重质量与角度、实测转速、相位基准约定）。录入即校验转速偏差与相位基准一致性。
3. **标定** `POST /api/batches/{id}/calibrate` — 幅相转复振动，试重前后差分，最小二乘拟合影响系数矩阵；返回系数（实部/虚部/幅值/相位）、逐运行逐测点拟合残差、条件数与测量来源（基线/试重运行 id、测点、校正面、时间）。
4. **求解** `POST /api/batches/{id}/solutions` — 先算连续配重（质量上限约束的投影最小二乘），再枚举孔位×配重规格的可安装组合（每面按与连续解接近度剪枝），按 **预测残振 → 总配重 → 公差最差结果** 字典序排序返回。
5. **复测** `POST /api/solutions/{id}/verifications` — 录入实测振动，返回逐测点预测/实测偏差、矢量偏差与相对基线的降幅比。
6. **导出** `GET /api/batches/{id}/export` — 完整 JSON 平衡记录（配置、运行、标定、方案、复测）。

## 计算模型

- **复振动**：`z = A·exp(iφ)`，相位单位为度；同一批次所有运行须使用同一相位基准约定。
- **影响系数**：`ΔV = T·αᵀ` 最小二乘，`T` 为各次试重的复配重矩阵；列归一化条件数 > 10³ 判病态。
- **连续解**：`min ‖αw + v0‖`，超限面按角度保持、幅值钳到上限后固定，迭代重解。
- **离散搜索**：每面孔位（每孔至多一块）× 配重规格 × 至多 `max_weights_per_plane` 块，总安装质量 ≤ 单面上限；两面候选笛卡尔组合评估。
- **最差情形**（一阶三角不等式界）：
  `wc = |pred| + Σ_p |α_p|·2·sin(δ/2)·M_p + A0·(ε_a + 2·sin(ε_φ/2))`
  （安装角公差 δ、面安装质量 M_p、幅值相对误差 ε_a、相位误差 ε_φ、基线振幅 A0）。

## 错误响应（422，指出对应运行与约束）

| error | 含义 | details |
|---|---|---|
| `speed_deviation` | 转速偏差超限 | 基准转速、允许偏差、超限运行 id 与偏差值 |
| `phase_reference_conflict` | 相位基准不一致 | 各约定对应的运行 id |
| `insufficient_trials` | 试重不足/秩亏 | 缺试重的校正面或秩信息 |
| `ill_conditioned_matrix` | 试重矩阵病态 | 条件数与阈值、相关运行 id |
| `no_feasible_combination` | 无可安装组合 | 校正面、质量上限、最小配重规格 |

## 目录

```
app/
  main.py            FastAPI 入口（lifespan 建表、领域错误 -> 422）
  database.py        SQLite 引擎/会话
  models.py          Batch / Run / Calibration / Solution / Verification
  schemas.py         Pydantic 请求响应
  services.py        校验、标定、求解、复测、导出编排
  core/
    vibration.py     幅相 <-> 复数
    influence.py     影响系数差分拟合
    correction.py    连续配重（约束投影最小二乘）
    discrete.py      离散组合枚举、剪枝、最差情形评估
  routers/           batches.py / solutions.py
tests/test_balancing.py   合成转子端到端 + 全部错误分支
```
