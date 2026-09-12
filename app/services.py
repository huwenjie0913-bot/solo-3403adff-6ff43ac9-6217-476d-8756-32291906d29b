"""业务编排：校验、标定、求解、复测、导出，以及慢转轴跳补偿。"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from .core.correction import solve_continuous
from .core.discrete import search_discrete_solutions, worst_case_residual
from .core.influence import fit_influence_coefficients
from .core.runout import (
    build_runout_summary,
    compensate_vectors,
)
from .core.vibration import amp_phase_to_complex, complex_to_amp_phase
from .errors import (
    CalibrationMissing,
    DomainError,
    InsufficientTrials,
    NotFoundError,
    PhaseReferenceConflict,
    RunoutPhaseReferenceConflict,
    RunoutProfileInvalid,
    SpeedDeviationError,
)
from .models import (
    Batch,
    Calibration,
    Run,
    RunoutProfile,
    RunoutRecord,
    Solution,
    Verification,
)
from .schemas import (
    BatchCreate,
    RunCreate,
    RunoutProfileCreate,
    RunoutRecordCreate,
    SolveRequest,
    VerificationCreate,
)


# ---------------------------------------------------------------- 工具


def get_batch_or_404(db: Session, batch_id: int) -> Batch:
    batch = db.get(Batch, batch_id)
    if batch is None:
        raise NotFoundError(f"批次 {batch_id} 不存在", {"batch_id": batch_id})
    return batch


def _sensor_names(batch: Batch) -> list[str]:
    return [s["name"] for s in batch.sensors]


def _plane_names(batch: Batch) -> list[str]:
    return [p["name"] for p in batch.planes]


def _run_vector(run: Run, sensors: list[str]) -> np.ndarray:
    by_sensor = {m["sensor"]: m for m in run.measurements}
    return np.array(
        [amp_phase_to_complex(by_sensor[s]["amplitude"], by_sensor[s]["phase"]) for s in sensors]
    )


def _check_speed(batch: Batch, runs: list[Run]) -> None:
    offenders = [
        {"run_id": r.id, "kind": r.kind, "speed": r.speed,
         "deviation": abs(r.speed - batch.reference_speed)}
        for r in runs
        if abs(r.speed - batch.reference_speed) > batch.speed_tolerance
    ]
    if offenders:
        raise SpeedDeviationError(
            "存在转速偏差超过允许值的运行",
            {
                "reference_speed": batch.reference_speed,
                "speed_tolerance": batch.speed_tolerance,
                "runs": offenders,
            },
        )


def _check_phase_reference(batch: Batch, runs: list[Run]) -> None:
    refs: dict[str, list[int]] = {}
    for r in runs:
        refs.setdefault(r.phase_reference, []).append(r.id)
    if len(refs) > 1:
        raise PhaseReferenceConflict(
            "批次内运行的相位基准不一致",
            {
                "references": {ref: ids for ref, ids in refs.items()},
                "hint": "同一批次的所有运行必须使用相同的相位基准约定",
            },
        )


# ---------------------------------------------------------------- 批次与运行


def create_batch(db: Session, payload: BatchCreate) -> Batch:
    batch = Batch(
        name=payload.name,
        description=payload.description,
        reference_speed=payload.reference_speed,
        speed_tolerance=payload.speed_tolerance,
        planes=[p.model_dump() for p in payload.planes],
        sensors=[s.model_dump() for s in payload.sensors],
        weight_specs=payload.weight_specs,
        amp_error=payload.amp_error,
        phase_error=payload.phase_error,
        angle_tolerance=payload.angle_tolerance,
    )
    db.add(batch)
    db.commit()
    db.refresh(batch)
    return batch


def add_run(db: Session, batch_id: int, payload: RunCreate) -> Run:
    batch = get_batch_or_404(db, batch_id)
    sensors = set(_sensor_names(batch))
    planes = set(_plane_names(batch))

    measured = [m.sensor for m in payload.measurements]
    unknown = [s for s in measured if s not in sensors]
    missing = [s for s in sensors if s not in measured]
    if unknown or missing or len(measured) != len(set(measured)):
        raise DomainError(
            "测点与批次配置不符",
            {"unknown_sensors": unknown, "missing_sensors": missing,
             "expected_sensors": sorted(sensors)},
        )
    bad_planes = [t.plane for t in payload.trial_weights if t.plane not in planes]
    if bad_planes:
        raise DomainError(
            "试重校正面与批次配置不符",
            {"unknown_planes": sorted(set(bad_planes)), "expected_planes": sorted(planes)},
        )

    run = Run(
        batch_id=batch.id,
        kind=payload.kind,
        speed=payload.speed,
        phase_reference=payload.phase_reference,
        measurements=[m.model_dump() for m in payload.measurements],
        trial_weights=[t.model_dump() for t in payload.trial_weights],
        note=payload.note,
    )
    # 录入即校验转速偏差与相位基准，并指出冲突运行
    _check_speed(batch, [run])
    _check_phase_reference(batch, list(batch.runs) + [run])

    db.add(run)
    db.commit()
    db.refresh(run)
    return run


# ---------------------------------------------------------------- 慢转轴跳档案


def _record_dicts(profile: RunoutProfile) -> list[dict]:
    return [
        {
            "id": r.id,
            "speed": r.speed,
            "phase_reference": r.phase_reference,
            "measurements": r.measurements,
        }
        for r in profile.records
    ]


def _profile_issues(profile: RunoutProfile, records: list[dict],
                    sensors: list[str]) -> list[dict]:
    """计算档案有效性问题：超速、测点缺失/未知/重复、基准不一致、离散度超限。"""
    issues: list[dict] = []
    sensor_set = set(sensors)

    if len(records) < 1:
        issues.append({
            "code": "insufficient_records",
            "record_count": 0,
            "required": 1,
        })

    # 记录超速
    over = [
        {"record_id": r["id"], "speed": r["speed"],
         "excess": float(r["speed"] - profile.slow_roll_speed_limit)}
        for r in records
        if r["speed"] > profile.slow_roll_speed_limit
    ]
    if over:
        issues.append({
            "code": "overspeed",
            "slow_roll_speed_limit": profile.slow_roll_speed_limit,
            "records": over,
        })

    # 逐记录测点缺失/未知/重复，并汇总从未出现过的测点
    coverage: list[dict] = []
    seen_ever: set[str] = set()
    for r in records:
        names = [m["sensor"] for m in r["measurements"]]
        counts = {n: names.count(n) for n in names}
        missing = [s for s in sensors if s not in counts]
        unknown = sorted({n for n in names if n not in sensor_set})
        duplicate = sorted({n for n, c in counts.items() if c > 1})
        if missing or unknown or duplicate:
            coverage.append({
                "record_id": r["id"],
                "missing_sensors": missing,
                "unknown_sensors": unknown,
                "duplicate_sensors": duplicate,
            })
        seen_ever.update(n for n in names if n in sensor_set)
    without_records = [s for s in sensors if s not in seen_ever]
    if coverage or without_records:
        issues.append({
            "code": "sensor_coverage",
            "records": coverage,
            "sensors_without_records": without_records,
            "expected_sensors": sensors,
        })

    # 档案内部相位基准一致性
    refs: dict[str, list[int]] = {}
    for r in records:
        refs.setdefault(r["phase_reference"], []).append(r["id"])
    if len(refs) > 1:
        issues.append({
            "code": "phase_reference_conflict",
            "references": {ref: ids for ref, ids in refs.items()},
        })

    # 逐测点重复测量离散度
    summary = build_runout_summary(records, sensors)
    bad_disp = [
        {
            "sensor": st.sensor,
            "dispersion": st.dispersion,
            "relative_dispersion": st.relative_dispersion,
            "record_count": st.count,
        }
        for st in summary.sensors
        if st.dispersion > profile.dispersion_limit
    ]
    if bad_disp:
        issues.append({
            "code": "dispersion_exceeded",
            "dispersion_limit": profile.dispersion_limit,
            "sensors": bad_disp,
        })

    return issues


def _recompute_profile(db: Session, profile: RunoutProfile) -> None:
    """依据当前记录重算逐测点统计与有效性问题并持久化。"""
    sensors = _sensor_names(profile.batch)
    records = _record_dicts(profile)
    profile.summary = build_runout_summary(records, sensors).as_json()
    profile.issues = _profile_issues(profile, records, sensors)
    db.flush()


def _get_profile_or_404(db: Session, batch: Batch, profile_id: int) -> RunoutProfile:
    profile = db.get(RunoutProfile, profile_id)
    if profile is None or profile.batch_id != batch.id:
        raise NotFoundError(
            f"轴跳档案 {profile_id} 不存在",
            {"runout_profile_id": profile_id, "batch_id": batch.id},
        )
    return profile


def create_runout_profile(
    db: Session, batch_id: int, payload: RunoutProfileCreate
) -> RunoutProfile:
    batch = get_batch_or_404(db, batch_id)
    profile = RunoutProfile(
        batch_id=batch.id,
        name=payload.name,
        slow_roll_speed_limit=payload.slow_roll_speed_limit,
        dispersion_limit=payload.dispersion_limit,
        summary={},
        issues=[],
        note=payload.note,
    )
    db.add(profile)
    db.flush()
    for rec in payload.records:
        db.add(RunoutRecord(
            profile_id=profile.id,
            speed=rec.speed,
            phase_reference=rec.phase_reference,
            measurements=[m.model_dump() for m in rec.measurements],
            note=rec.note,
        ))
    db.flush()
    _recompute_profile(db, profile)
    db.commit()
    db.refresh(profile)
    return profile


def add_runout_record(
    db: Session, batch_id: int, profile_id: int, payload: RunoutRecordCreate
) -> RunoutProfile:
    batch = get_batch_or_404(db, batch_id)
    profile = _get_profile_or_404(db, batch, profile_id)
    db.add(RunoutRecord(
        profile_id=profile.id,
        speed=payload.speed,
        phase_reference=payload.phase_reference,
        measurements=[m.model_dump() for m in payload.measurements],
        note=payload.note,
    ))
    db.flush()
    _recompute_profile(db, profile)
    db.commit()
    db.refresh(profile)
    return profile


def _ensure_profile_usable(profile: RunoutProfile) -> None:
    """档案存在结构性/限值问题时拒绝使用，并指出对应记录与测点。"""
    if profile.issues:
        raise RunoutProfileInvalid(
            f"轴跳档案「{profile.name}」未通过校验，拒绝用于补偿",
            {
                "runout_profile_id": profile.id,
                "runout_profile_name": profile.name,
                "issues": profile.issues,
            },
        )


def _runout_mean_vector(profile: RunoutProfile, sensors: list[str]) -> np.ndarray:
    return np.array([
        complex(
            profile.summary[s]["real"],
            profile.summary[s]["imag"],
        )
        for s in sensors
    ])


def _resolve_runout_profile(
    db: Session, batch: Batch, profile_id: int | None, runs: list[Run]
) -> tuple[RunoutProfile | None, np.ndarray]:
    """取出并校验轴跳档案，返回 (档案, 与批次测点对齐的轴跳复矢量)。"""
    sensors = _sensor_names(batch)
    if profile_id is None:
        return None, np.zeros(len(sensors), dtype=complex)

    profile = _get_profile_or_404(db, batch, profile_id)
    _recompute_profile(db, profile)  # 用前按最新记录重算，问题不入库改动也要提交
    db.commit()
    _ensure_profile_usable(profile)

    # 档案内部基准须唯一（issues 已保证），且与被补偿运行的基准一致
    record_refs = {r.phase_reference for r in profile.records}
    run_refs = {r.phase_reference for r in runs}
    if record_refs != run_refs:
        raise RunoutPhaseReferenceConflict(
            "轴跳档案与运行的相位基准约定不一致，不能扣除轴跳",
            {
                "runout_profile_id": profile.id,
                "runout_phase_references": sorted(record_refs),
                "run_phase_references": sorted(run_refs),
                "run_ids": [r.id for r in runs],
                "record_ids": [r.id for r in profile.records],
            },
        )
    return profile, _runout_mean_vector(profile, sensors)


def _profile_block(profile: RunoutProfile, phase_reference: str | None) -> dict:
    """档案快照（写入标定/方案/复测与导出，切换或修改档案不影响既得结果）。"""
    return {
        "runout_profile_id": profile.id,
        "runout_profile_name": profile.name,
        "slow_roll_speed_limit": profile.slow_roll_speed_limit,
        "dispersion_limit": profile.dispersion_limit,
        "phase_reference": phase_reference,
        "summary": profile.summary,
        "snapshotted_at": datetime.now(timezone.utc).isoformat(),
    }


def _compensation_json(comps, dispersion_by_sensor: dict[str, float]) -> tuple[dict, list[str]]:
    """把逐测点补偿结果转为响应 JSON，返回 (按测点字典, 不可判定测点)。"""
    out: dict[str, dict] = {}
    undecidable: list[str] = []
    for c in comps:
        out[c.sensor] = {
            "raw": {"amplitude": c.raw_amplitude, "phase": c.raw_phase},
            "compensation": {
                "amplitude": c.compensation_amplitude,
                "phase": c.compensation_phase,
            },
            "net": {"amplitude": c.net_amplitude, "phase": c.net_phase},
            "dispersion": float(dispersion_by_sensor.get(c.sensor, 0.0)),
            "resolution": c.resolution,
            "undecidable": c.undecidable,
        }
        if c.undecidable:
            undecidable.append(c.sensor)
    return out, undecidable


# ---------------------------------------------------------------- 标定


def calibrate(
    db: Session, batch_id: int, runout_profile_id: int | None = None
) -> Calibration:
    batch = get_batch_or_404(db, batch_id)

    # 档案无效（测点缺失、超速、离散度超限等）时优先拒绝，
    # 不要求批次先具备完整运行
    if runout_profile_id is not None:
        profile_spec = _get_profile_or_404(db, batch, runout_profile_id)
        _recompute_profile(db, profile_spec)
        db.commit()
        _ensure_profile_usable(profile_spec)

    runs = list(batch.runs)
    _check_speed(batch, runs)
    _check_phase_reference(batch, runs)

    baselines = [r for r in runs if r.kind == "baseline"]
    trials = [r for r in runs if r.kind == "trial"]
    if not baselines:
        raise InsufficientTrials("缺少基线运行", {"missing": "baseline"})
    if len(baselines) > 1:
        raise DomainError(
            "批次中存在多条基线运行，无法确定差分基准",
            {"baseline_run_ids": [r.id for r in baselines]},
        )
    if not trials:
        raise InsufficientTrials("缺少试重运行", {"missing": "trial_runs"})

    sensors = _sensor_names(batch)
    planes = _plane_names(batch)

    # 先扣除慢转轴跳，再做差分拟合（同一均值对各运行作差时会抵消，系数不变，
    # 基线变为净振动；档案与运行相位基准不一致时直接拒绝）
    profile, runout_vec = _resolve_runout_profile(
        db, batch, runout_profile_id, [baselines[0]] + trials
    )

    raw_baseline = _run_vector(baselines[0], sensors)
    raw_trials = np.array([_run_vector(r, sensors) for r in trials])
    baseline_vec = raw_baseline - runout_vec
    trial_vectors = raw_trials - runout_vec[np.newaxis, :]

    trial_matrix = np.zeros((len(trials), len(planes)), dtype=complex)
    for k, run in enumerate(trials):
        for tw in run.trial_weights:
            p = planes.index(tw["plane"])
            trial_matrix[k, p] += amp_phase_to_complex(tw["mass"], tw["angle"])

    result = fit_influence_coefficients(
        baseline_vec,
        trial_vectors,
        trial_matrix,
        run_ids=[r.id for r in trials],
        plane_names=planes,
    )

    coefficients = {
        s: {
            p: {
                "real": float(result.coefficients[i, j].real),
                "imag": float(result.coefficients[i, j].imag),
                "magnitude": float(abs(result.coefficients[i, j])),
                "phase": float(np.rad2deg(np.angle(result.coefficients[i, j])) % 360.0),
            }
            for j, p in enumerate(planes)
        }
        for i, s in enumerate(sensors)
    }
    residuals = []
    for k, run in enumerate(trials):
        for i, s in enumerate(sensors):
            amp, phase = complex_to_amp_phase(result.residuals[k, i])
            residuals.append(
                {
                    "run_id": run.id,
                    "sensor": s,
                    "amplitude": amp,
                    "phase": phase,
                    "relative": float(result.relative_residuals[k, i]),
                }
            )
    provenance = {
        "baseline_run_id": baselines[0].id,
        "trial_run_ids": [r.id for r in trials],
        "sensors": sensors,
        "planes": planes,
        "reference_speed": batch.reference_speed,
        "phase_reference": baselines[0].phase_reference,
        "runout_profile_id": profile.id if profile else None,
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }

    # 轴跳补偿快照：基线 + 各次试重逐测点的原始值/补偿量/净振动
    runout_compensation = None
    if profile is not None:
        dispersion_by = {s: profile.summary[s]["dispersion"] for s in sensors}
        run_items = []
        all_undecidable: set[str] = set()
        for run, raw_vec in [(baselines[0], raw_baseline), *zip(trials, raw_trials)]:
            comps = compensate_vectors(
                raw_vec, runout_vec, sensors,
                batch.amp_error, batch.phase_error,
            )
            comp_json, und = _compensation_json(comps, dispersion_by)
            all_undecidable.update(und)
            run_items.append({
                "run_id": run.id, "kind": run.kind, "sensors": comp_json,
            })
        runout_compensation = {
            **_profile_block(profile, baselines[0].phase_reference),
            "runs": run_items,
            "undecidable_sensors": sorted(all_undecidable),
        }

    # 同一档案（含无档案）的旧标定被本次取代；切换档案不删除其它标定
    for old in list(batch.calibrations):
        if old.runout_profile_id == runout_profile_id:
            db.delete(old)
    db.flush()

    cal = Calibration(
        batch_id=batch.id,
        runout_profile_id=profile.id if profile else None,
        coefficients=coefficients,
        residuals=residuals,
        condition=result.condition,
        provenance=provenance,
        runout_compensation=runout_compensation,
    )
    db.add(cal)
    db.commit()
    db.refresh(cal)
    return cal


def _coefficient_matrix(cal: Calibration, sensors: list[str], planes: list[str]) -> np.ndarray:
    return np.array(
        [
            [complex(cal.coefficients[s][p]["real"], cal.coefficients[s][p]["imag"]) for p in planes]
            for s in sensors
        ]
    )


# ---------------------------------------------------------------- 求解


def solve(db: Session, batch_id: int, req: SolveRequest) -> tuple[Solution, list[Solution]]:
    batch = get_batch_or_404(db, batch_id)
    profile_id = req.runout_profile_id

    # 使用与指定档案匹配的标定；没有则按该档案标定（自动扣除轴跳）。
    # 切换档案不会改写已有的其它标定与方案。
    cal = next(
        (
            c
            for c in reversed(batch.calibrations)
            if c.runout_profile_id == profile_id
        ),
        None,
    )
    if cal is None:
        cal = calibrate(db, batch_id, profile_id)
    if cal is None:  # pragma: no cover - 防御
        raise CalibrationMissing("批次尚未标定")

    sensors = _sensor_names(batch)
    planes = _plane_names(batch)
    alpha = _coefficient_matrix(cal, sensors, planes)

    baselines = [r for r in batch.runs if r.kind == "baseline"]
    raw_v0 = _run_vector(baselines[0], sensors)
    if profile_id is not None:
        profile = _get_profile_or_404(db, batch, profile_id)
        _recompute_profile(db, profile)
        db.commit()
        _ensure_profile_usable(profile)
        runout_vec = _runout_mean_vector(profile, sensors)
    else:
        profile = None
        runout_vec = np.zeros(len(sensors), dtype=complex)
    v0 = raw_v0 - runout_vec

    limits = [float(p["mass_limit"]) for p in batch.planes]
    cont = solve_continuous(alpha, v0, limits)

    discrete = search_discrete_solutions(
        alpha=alpha,
        v0=v0,
        continuous_target=cont.weights,
        plane_names=planes,
        hole_angles=[p["hole_angles"] for p in batch.planes],
        weight_specs=[float(m) for m in batch.weight_specs],
        mass_limits=limits,
        amp_error=batch.amp_error,
        phase_error_deg=batch.phase_error,
        angle_tolerance_deg=batch.angle_tolerance,
        max_weights_per_plane=req.max_weights_per_plane,
        keep_per_plane=req.keep_per_plane,
        top=req.top,
    )

    def _residual_json(pred: np.ndarray) -> dict:
        out = {}
        for i, s in enumerate(sensors):
            amp, phase = complex_to_amp_phase(pred[i])
            out[s] = {"amplitude": amp, "phase": phase}
        return out

    # 净残振可分辨范围：净基线由“运行测量 − 慢转均值”得到，
    # 不确定半径为两次幅相测量误差界之和；落入范围内即不可判定。
    k_err = batch.amp_error + 2.0 * float(
        np.sin(np.deg2rad(batch.phase_error) / 2.0)
    )

    def _resolution_json(pred: np.ndarray) -> tuple[dict, list[str]]:
        per_sensor, und = {}, []
        for i, s in enumerate(sensors):
            res = float((abs(raw_v0[i]) + abs(runout_vec[i])) * k_err)
            pred_amp = float(abs(pred[i]))
            flag = bool(pred_amp <= res)
            per_sensor[s] = {
                "resolution": res,
                "predicted_amplitude": pred_amp,
                "undecidable": flag,
            }
            if flag:
                und.append(s)
        return {"sensors": per_sensor, "undecidable_sensors": und}, und

    baseline_compensation = None
    if profile is not None:
        comps = compensate_vectors(
            raw_v0, runout_vec, sensors, batch.amp_error, batch.phase_error
        )
        dispersion_by = {s: profile.summary[s]["dispersion"] for s in sensors}
        comp_json, und_baseline = _compensation_json(comps, dispersion_by)
        baseline_compensation = {
            **_profile_block(profile, baselines[0].phase_reference),
            "baseline_run_id": baselines[0].id,
            "sensors": comp_json,
            "undecidable_sensors": und_baseline,
        }

    # 仅取代“同一档案选择”且未复测的旧方案；已复测方案与其它档案的方案保留
    for old in list(batch.solutions):
        if not old.verifications and old.runout_profile_id == profile_id:
            db.delete(old)
    db.flush()

    pid = profile.id if profile else None

    def _build_solution(kind: str, rank: int, weights_json, pred, total_mass,
                        masses_per_plane, assignments_worst_case=None) -> Solution:
        res_json, _ = _resolution_json(pred)
        return Solution(
            batch_id=batch.id,
            runout_profile_id=pid,
            kind=kind,
            rank=rank,
            weights=weights_json,
            predicted_residual=_residual_json(pred),
            predicted_metric=float(np.max(np.abs(pred))),
            total_mass=float(total_mass),
            worst_case=worst_case_residual(
                pred,
                alpha,
                masses_per_plane,
                np.abs(v0),
                batch.amp_error,
                batch.phase_error,
                batch.angle_tolerance,
            ),
            resolution=res_json,
            runout_compensation=baseline_compensation,
        )

    cont_sol = _build_solution(
        kind="continuous",
        rank=0,
        weights_json=[
            {
                "plane": planes[p],
                "mass": float(abs(cont.weights[p])),
                "angle": float(np.rad2deg(np.angle(cont.weights[p])) % 360.0),
                "assignments": [],
            }
            for p in range(len(planes))
        ],
        pred=cont.predicted_residual,
        total_mass=np.sum(np.abs(cont.weights)),
        masses_per_plane=[float(abs(w)) for w in cont.weights],
    )
    db.add(cont_sol)
    db.flush()

    discrete_sols: list[Solution] = []
    for rank, d in enumerate(discrete, start=1):
        weights = []
        for p, plane in enumerate(planes):
            plane_assign = [a for a in d.assignments if a.plane == plane]
            weights.append(
                {
                    "plane": plane,
                    "mass": float(abs(d.weights[p])),
                    "angle": float(np.rad2deg(np.angle(d.weights[p])) % 360.0),
                    "assignments": [
                        {"hole_angle": a.hole_angle, "mass": a.mass} for a in plane_assign
                    ],
                }
            )
        sol = _build_solution(
            kind="discrete",
            rank=rank,
            weights_json=weights,
            pred=d.predicted_residual,
            total_mass=d.total_mass,
            masses_per_plane=[
                float(sum(a.mass for a in d.assignments if a.plane == plane))
                for plane in planes
            ],
        )
        db.add(sol)
        discrete_sols.append(sol)
    db.commit()
    db.refresh(cont_sol)
    for s in discrete_sols:
        db.refresh(s)
    return cont_sol, discrete_sols


# ---------------------------------------------------------------- 复测


def add_verification(
    db: Session, solution_id: int, payload: VerificationCreate
) -> Verification:
    sol = db.get(Solution, solution_id)
    if sol is None:
        raise NotFoundError(f"方案 {solution_id} 不存在", {"solution_id": solution_id})
    batch = sol.batch
    sensors = _sensor_names(batch)

    measured = [m.sensor for m in payload.measurements]
    if set(measured) != set(sensors) or len(measured) != len(set(measured)):
        raise DomainError(
            "复测测点与批次配置不符",
            {"expected_sensors": sensors, "received": measured},
        )
    if abs(payload.speed - batch.reference_speed) > batch.speed_tolerance:
        raise SpeedDeviationError(
            "复测转速偏差超过允许值",
            {
                "reference_speed": batch.reference_speed,
                "speed_tolerance": batch.speed_tolerance,
                "runs": [{"speed": payload.speed,
                          "deviation": abs(payload.speed - batch.reference_speed)}],
            },
        )

    baselines = [r for r in batch.runs if r.kind == "baseline"]
    baseline_run = baselines[0] if baselines else None

    # 复测相位基准：显式给定 > 基线约定；须与方案/基线一致
    expected_ref = baseline_run.phase_reference if baseline_run else None
    phase_ref = payload.phase_reference or expected_ref
    if expected_ref is not None and phase_ref != expected_ref:
        raise PhaseReferenceConflict(
            "复测相位基准与批次运行不一致",
            {
                "references": {
                    expected_ref: [baseline_run.id] if baseline_run else [],
                    phase_ref: ["verification"],
                },
            },
        )

    # 轴跳档案：显式指定 > 沿用方案所用档案；先扣除再与方案预测净残振对比
    profile_id = (
        payload.runout_profile_id
        if payload.runout_profile_id is not None
        else sol.runout_profile_id
    )
    runout_vec = np.zeros(len(sensors), dtype=complex)
    profile = None
    if profile_id is not None:
        profile = _get_profile_or_404(db, batch, profile_id)
        _recompute_profile(db, profile)
        db.commit()
        _ensure_profile_usable(profile)
        record_refs = {r.phase_reference for r in profile.records}
        if phase_ref is not None and record_refs != {phase_ref}:
            raise RunoutPhaseReferenceConflict(
                "轴跳档案与复测的相位基准约定不一致，不能扣除轴跳",
                {
                    "runout_profile_id": profile.id,
                    "runout_phase_references": sorted(record_refs),
                    "verification_phase_reference": phase_ref,
                    "record_ids": [r.id for r in profile.records],
                },
            )
        runout_vec = _runout_mean_vector(profile, sensors)

    by_sensor = {m.sensor: m for m in payload.measurements}
    raw_vec = np.array([
        amp_phase_to_complex(by_sensor[s].amplitude, by_sensor[s].phase)
        for s in sensors
    ])
    comps = compensate_vectors(
        raw_vec, runout_vec, sensors, batch.amp_error, batch.phase_error
    )

    # 基线净振幅：方案扣过速跳时取其补偿快照中的净振动，否则用原始基线
    baseline_net_amp: dict[str, float] = {}
    if baseline_run is not None:
        raw_base = {m["sensor"]: m for m in baseline_run.measurements}
        block = sol.runout_compensation
        for s in sensors:
            if block and s in block["sensors"]:
                baseline_net_amp[s] = block["sensors"][s]["net"]["amplitude"]
            else:
                baseline_net_amp[s] = raw_base[s]["amplitude"]

    dispersion_by = (
        {s: profile.summary[s]["dispersion"] for s in sensors} if profile else {}
    )
    comp_json, undecidable_sensors = _compensation_json(comps, dispersion_by)

    comparison: dict = {
        "sensors": {},
        "undecidable_sensors": undecidable_sensors,
        "balance_verdict": "undecidable" if undecidable_sensors else "decidable",
    }
    if profile is not None:
        comparison["runout_profile"] = _profile_block(profile, phase_ref)

    for i, c in enumerate(comps):
        pred = sol.predicted_residual[c.sensor]
        z_pred = amp_phase_to_complex(pred["amplitude"], pred["phase"])
        net = raw_vec[i] - runout_vec[i]
        dev = abs(net - z_pred)
        rel = float(dev / max(abs(net), 1e-12))
        entry = {
            "predicted": {"amplitude": pred["amplitude"], "phase": pred["phase"]},
            "raw_measured": {"amplitude": c.raw_amplitude, "phase": c.raw_phase},
            "runout_compensation": {
                "amplitude": c.compensation_amplitude,
                "phase": c.compensation_phase,
            },
            "measured": {"amplitude": c.net_amplitude, "phase": c.net_phase},
            "resolution": c.resolution,
            "undecidable": c.undecidable,
            "verdict": "undecidable" if c.undecidable else "decidable",
            "vector_deviation": float(dev),
            "relative_deviation": rel,
        }
        base_amp = baseline_net_amp.get(c.sensor)
        if base_amp is not None and base_amp > 0:
            entry["baseline_amplitude"] = base_amp
            # 净振动不可分辨时降幅无意义：置空，禁止据此判定平衡达标
            entry["reduction_ratio"] = (
                None if c.undecidable else float(1.0 - c.net_amplitude / base_amp)
            )
        comparison["sensors"][c.sensor] = entry

    ver = Verification(
        solution_id=sol.id,
        runout_profile_id=profile.id if profile else None,
        speed=payload.speed,
        measurements=[m.model_dump() for m in payload.measurements],
        comparison=comparison,
        note=payload.note,
    )
    db.add(ver)
    db.commit()
    db.refresh(ver)
    return ver


# ---------------------------------------------------------------- 导出


def _profile_export(profile: RunoutProfile) -> dict:
    return {
        "id": profile.id,
        "name": profile.name,
        "slow_roll_speed_limit": profile.slow_roll_speed_limit,
        "dispersion_limit": profile.dispersion_limit,
        "summary": profile.summary,
        "issues": profile.issues,
        "usable": not profile.issues,
        "note": profile.note,
        "created_at": profile.created_at.isoformat(),
        "records": [
            {
                "id": r.id,
                "speed": r.speed,
                "phase_reference": r.phase_reference,
                "measurements": r.measurements,
                "note": r.note,
                "created_at": r.created_at.isoformat(),
            }
            for r in profile.records
        ],
    }


def export_batch(db: Session, batch_id: int) -> dict:
    batch = get_batch_or_404(db, batch_id)

    # 档案实际使用位置：标定 / 方案 / 复测
    usage: dict[str, dict] = {}
    for p in batch.runout_profiles:
        usage[str(p.id)] = {
            "runout_profile_id": p.id,
            "runout_profile_name": p.name,
            "calibration_ids": [],
            "solution_ids": [],
            "verification_ids": [],
        }
    for c in batch.calibrations:
        if c.runout_profile_id and str(c.runout_profile_id) in usage:
            usage[str(c.runout_profile_id)]["calibration_ids"].append(c.id)
    for s in batch.solutions:
        if s.runout_profile_id and str(s.runout_profile_id) in usage:
            usage[str(s.runout_profile_id)]["solution_ids"].append(s.id)
        for v in s.verifications:
            if v.runout_profile_id and str(v.runout_profile_id) in usage:
                usage[str(v.runout_profile_id)]["verification_ids"].append(v.id)

    return {
        "record_type": "two_plane_field_balancing",
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "batch": {
            "id": batch.id,
            "name": batch.name,
            "description": batch.description,
            "reference_speed": batch.reference_speed,
            "speed_tolerance": batch.speed_tolerance,
            "planes": batch.planes,
            "sensors": batch.sensors,
            "weight_specs": batch.weight_specs,
            "tolerances": {
                "amp_error": batch.amp_error,
                "phase_error": batch.phase_error,
                "angle_tolerance": batch.angle_tolerance,
            },
            "created_at": batch.created_at.isoformat(),
        },
        "runs": [
            {
                "id": r.id,
                "kind": r.kind,
                "speed": r.speed,
                "phase_reference": r.phase_reference,
                "measurements": r.measurements,
                "trial_weights": r.trial_weights,
                "note": r.note,
                "created_at": r.created_at.isoformat(),
            }
            for r in batch.runs
        ],
        "runout_profiles": [_profile_export(p) for p in batch.runout_profiles],
        "runout_usage": list(usage.values()),
        "calibrations": [
            {
                "id": c.id,
                "runout_profile_id": c.runout_profile_id,
                "coefficients": c.coefficients,
                "residuals": c.residuals,
                "condition": c.condition,
                "provenance": c.provenance,
                "runout_compensation": c.runout_compensation,
                "created_at": c.created_at.isoformat(),
            }
            for c in batch.calibrations
        ],
        "calibration": (
            {
                "id": batch.calibration.id,
                "runout_profile_id": batch.calibration.runout_profile_id,
                "coefficients": batch.calibration.coefficients,
                "residuals": batch.calibration.residuals,
                "condition": batch.calibration.condition,
                "provenance": batch.calibration.provenance,
                "runout_compensation": batch.calibration.runout_compensation,
                "created_at": batch.calibration.created_at.isoformat(),
            }
            if batch.calibration
            else None
        ),
        "solutions": [
            {
                "id": s.id,
                "runout_profile_id": s.runout_profile_id,
                "kind": s.kind,
                "rank": s.rank,
                "weights": s.weights,
                "predicted_residual": s.predicted_residual,
                "predicted_metric": s.predicted_metric,
                "total_mass": s.total_mass,
                "worst_case": s.worst_case,
                "resolution": s.resolution,
                "runout_compensation": s.runout_compensation,
                "created_at": s.created_at.isoformat(),
                "verifications": [
                    {
                        "id": v.id,
                        "runout_profile_id": v.runout_profile_id,
                        "speed": v.speed,
                        "measurements": v.measurements,
                        "comparison": v.comparison,
                        "note": v.note,
                        "created_at": v.created_at.isoformat(),
                    }
                    for v in s.verifications
                ],
            }
            for s in batch.solutions
        ],
    }
