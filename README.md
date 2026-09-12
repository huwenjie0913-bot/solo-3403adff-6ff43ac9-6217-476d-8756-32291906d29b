# 双面现场动平衡 API

面向旋转设备检修人员的双面现场动平衡 REST API。接收两个校正面、振动测点、基准转速以及基线/各次试重的幅相数据，按试重前后差分拟合影响系数矩阵，计算连续配重并搜索可安装的离散配重组合，支持复测对比与 JSON 平衡记录导出。支持为批次建立**慢转轴跳档案**，在标定、求解、复测前先逐测点扣除轴跳复矢量。

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
3. **（可选）建立慢转轴跳档案** `POST /api/batches/{id}/runout-profiles` — 设置慢转转速上限与重复测量离散度阈值，随档案或经 `.../runout-profiles/{pid}/records` 录入一组慢转记录（转速、相位基准、各测点 1X 振幅/相位）。系统把同一测点的记录换算为复矢量，给出**复均值、RMS 离散度、相对离散度与记录来源**；档案即使存在校验问题也会保存并列出问题，但不可用于补偿。
4. **标定** `POST /api/batches/{id}/calibrate` — 可带 `{"runout_profile_id": N}`，先逐测点扣除轴跳再幅相转复振动、试重前后差分、最小二乘拟合影响系数矩阵；返回系数、逐运行逐测点拟合残差、条件数、测量来源，以及**逐运行逐测点的原始值/补偿量/净振动/可分辨范围**。同一轴跳选择（含无档案）重复标定取代旧标定；切换档案保留各自标定。
5. **求解** `POST /api/batches/{id}/solutions` — 请求可带 `runout_profile_id`；先扣除轴跳，再算连续配重（质量上限约束的投影最小二乘）并枚举孔位×配重规格的可安装组合，按 **预测残振 → 总配重 → 公差最差结果** 字典序排序。仅同一档案选择且未复测的旧方案被取代，其它方案保留。
6. **复测** `POST /api/solutions/{id}/verifications` — 默认沿用方案所用轴跳档案，也可用 `runout_profile_id` 显式指定；先扣除轴跳，再返回逐测点原始测量/轴跳补偿量/净振动、与预测的偏差、相对基线（净）降幅及不可判定标记。
7. **导出** `GET /api/batches/{id}/export` — 完整 JSON 平衡记录：配置、运行、轴跳档案快照及其在标定/方案/复测中的**实际使用位置**、标定（全部历史）、方案、复测；各使用点内嵌档案快照。

## 计算模型

- **复振动**：`z = A·exp(iφ)`，相位单位为度；同一批次所有运行须使用同一相位基准约定。
- **轴跳档案**：同一测点多条慢转记录取复均值 `mean_s = Σz/n`，离散度 `disp_s = sqrt(Σ|z_k−mean|²/n)`；使用前须通过：测点齐全且无重复/未知、全部记录 `speed ≤ 慢转转速上限`、档案内相位基准一致且与被补偿运行一致、逐测点 `disp_s ≤ 离散度阈值`。
- **轴跳补偿**：`z_net = z_raw − mean_s`。同一均值对试重差分抵消，影响系数不变，基线与残振变为净振动。
- **可分辨范围（不可判定）**：一次幅相测量的不确定半径 `u(z)=|z|·(ε_a+2·sin(ε_φ/2))`，净振动由运行测量与慢转均值相减得到，故 `res_s = u(raw_s)+u(mean_s)`。`|z_net| ≤ res_s` 时该测点标为 `undecidable`；复测中此类测点不输出降幅，整体 `balance_verdict=undecidable`，**不得当作平衡达标**。
- **影响系数**：`ΔV = T·αᵀ` 最小二乘，`T` 为各次试重的复配重矩阵；列归一化条件数 > 10³ 判病态。
- **连续解**：`min ‖αw + v0‖`（`v0` 为扣轴跳后的净基线），超限面按角度保持、幅值钳到上限后固定，迭代重解。
- **离散搜索**：每面孔位（每孔至多一块）× 配重规格 × 至多 `max_weights_per_plane` 块，总安装质量 ≤ 单面上限；两面候选笛卡尔组合评估。
- **最差情形**（一阶三角不等式界）：
  `wc = |pred| + Σ_p |α_p|·2·sin(δ/2)·M_p + A0·(ε_a + 2·sin(ε_φ/2))`
  （安装角公差 δ、面安装质量 M_p、幅值相对误差 ε_a、相位误差 ε_φ、净基线振幅 A0）。

## 错误响应（422，指出对应运行、记录与测点）

| error | 含义 | details |
|---|---|---|
| `speed_deviation` | 转速偏差超限 | 基准转速、允许偏差、超限运行 id 与偏差值 |
| `phase_reference_conflict` | 相位基准不一致 | 各约定对应的运行 id |
| `runout_phase_reference_conflict` | 轴跳档案与运行/复测基准不一致 | 档案与运行各自的基准约定、记录/运行 id |
| `insufficient_trials` | 试重不足/秩亏 | 缺试重的校正面或秩信息 |
| `ill_conditioned_matrix` | 试重矩阵病态 | 条件数与阈值、相关运行 id |
| `no_feasible_combination` | 无可安装组合 | 校正面、质量上限、最小配重规格 |
| `runout_profile_invalid` | 轴跳档案校验未过，拒绝使用 | 问题清单：`overspeed`（超速记录）、`sensor_coverage`（缺失/未知/重复测点及记录）、`dispersion_exceeded`（超限测点与离散度）、`phase_reference_conflict`、`insufficient_records` |

## 目录

```
app/
  main.py            FastAPI 入口（lifespan 建表、领域错误 -> 422）
  database.py        SQLite 引擎/会话
  models.py          Batch / Run / RunoutProfile / RunoutRecord /
                     Calibration / Solution / Verification
  schemas.py         Pydantic 请求响应
  services.py        校验、轴跳档案、补偿扣除、标定、求解、复测、导出编排
  core/
    vibration.py     幅相 <-> 复数
    runout.py        轴跳复矢量统计、逐测点补偿与可分辨范围
    influence.py     影响系数差分拟合
    correction.py    连续配重（约束投影最小二乘）
    discrete.py      离散组合枚举、剪枝、最差情形评估
  routers/           batches.py / runout.py / solutions.py
tests/test_balancing.py   合成转子端到端 + 全部错误分支
tests/test_runout.py      慢转轴跳建档、补偿、拒绝分支、不可判定、切换隔离、导出
```
