"""业务编排：校验、标定、求解、复测、导出，以及慢转轴跳补偿。"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from .core.correction import solve_continuous
from .core.discrete import search_discrete_solutions, worst_case_residual
from .core.influence import fit_influence_coefficients
from .core import mixed as mixed_core
from .core.runout import (
    build_runout_summary,
    compensate_vectors,
)
from .core.vibration import amp_phase_to_complex, complex_to_amp_phase, normalize_angle
from .errors import (
    CalibrationMissing,
    DomainError,
    InsufficientTrials,
    MixedCorrectionConfigError,
    NotFoundError,
    PhaseReferenceConflict,
    RunoutPhaseReferenceConflict,
    RunoutProfileInvalid,
    SpeedDeviationError,
)
from .models import (
    Batch,
    Calibration,
    MixedPlan,
    MixedVerification,
    Run,
    RunoutProfile,
    RunoutRecord,
    Solution,
    Verification,
)
from .schemas import (
    BatchCreate,
    MixedPlanCreate,
    MixedSearchRequest,
    MixedVerificationCreate,
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


# ---------------------------------------------------------------- 加重/去料混合校正


def _get_calibration(db: Session, batch: Batch, calibration_id: int) -> Calibration:
    cal = db.get(Calibration, calibration_id)
    if cal is None or cal.batch_id != batch.id:
        raise NotFoundError(
            f"标定 {calibration_id} 不存在",
            {"calibration_id": calibration_id, "batch_id": batch.id},
        )
    return cal


def _mixed_prepare(
    db: Session, batch: Batch, *, calibration_id: int | None,
    runout_profile_id: int | None,
) -> tuple[Calibration, RunoutProfile | None, np.ndarray, np.ndarray,
           np.ndarray, list[str], list[str], Run | None]:
    """解析混合校正共用的标定/档案/净基线，并检查相位基准一致。

    不新建、不改写任何既有标定；标定与档案选择必须显式匹配。
    """
    sensors = _sensor_names(batch)
    planes = _plane_names(batch)

    cal: Calibration | None = None
    if calibration_id is not None:
        cal = _get_calibration(db, batch, calibration_id)
        if runout_profile_id is not None and cal.runout_profile_id != runout_profile_id:
            raise MixedCorrectionConfigError(
                "指定标定与轴跳档案不匹配",
                {"calibration_id": cal.id,
                 "calibration_runout_profile_id": cal.runout_profile_id,
                 "requested_runout_profile_id": runout_profile_id},
            )
    else:
        cal = next(
            (c for c in reversed(batch.calibrations)
             if c.runout_profile_id == runout_profile_id),
            None,
        )
        if cal is None:
            raise CalibrationMissing(
                "没有与所选轴跳档案匹配的标定，请先标定（混合校正不自动新建标定）",
                {"runout_profile_id": runout_profile_id},
            )

    baselines = [r for r in batch.runs if r.kind == "baseline"]
    if not baselines:
        raise InsufficientTrials("缺少基线运行", {"missing": "baseline"})
    if len(baselines) > 1:
        raise DomainError(
            "批次中存在多条基线运行，无法确定差分基准",
            {"baseline_run_ids": [r.id for r in baselines]},
        )
    baseline_run = baselines[0]

    profile: RunoutProfile | None = None
    runout_vec = np.zeros(len(sensors), dtype=complex)
    if runout_profile_id is not None:
        profile = _get_profile_or_404(db, batch, runout_profile_id)
        _recompute_profile(db, profile)
        db.commit()
        _ensure_profile_usable(profile)
        record_refs = {r.phase_reference for r in profile.records}
        if record_refs != {baseline_run.phase_reference}:
            raise RunoutPhaseReferenceConflict(
                "轴跳档案与基线运行的相位基准约定不一致，不能扣除轴跳",
                {
                    "runout_profile_id": profile.id,
                    "runout_phase_references": sorted(record_refs),
                    "run_phase_references": [baseline_run.phase_reference],
                    "run_ids": [baseline_run.id],
                    "record_ids": [r.id for r in profile.records],
                },
            )
        runout_vec = _runout_mean_vector(profile, sensors)

    raw_v0 = _run_vector(baseline_run, sensors)
    v0 = raw_v0 - runout_vec
    alpha = _coefficient_matrix(cal, sensors, planes)
    return cal, profile, raw_v0, v0, runout_vec, sensors, planes, baseline_run


def _build_plane_specs(
    batch: Batch, planes: list[str], req_planes: list,
    max_weights_per_plane: int = 3,
) -> list[mixed_core.PlaneSpec]:
    """逐面校验几何归属并构造混合校正面规格（纯数值在 core 内再校验）。"""
    if len({p.plane for p in req_planes}) != len(req_planes):
        raise MixedCorrectionConfigError(
            "混合校正请求中存在重复校正面",
            {"planes": [p.plane for p in req_planes]},
        )
    by_name = {p.plane: p for p in req_planes}
    missing = [p for p in planes if p not in by_name]
    unknown = [p for p in by_name if p not in planes]
    if missing or unknown:
        raise MixedCorrectionConfigError(
            "混合校正校正面与批次配置不符",
            {"missing_planes": missing, "unknown_planes": unknown,
             "expected_planes": planes},
        )

    specs: list[mixed_core.PlaneSpec] = []
    for p in planes:
        cfg = next(c for c in batch.planes if c["name"] == p)
        req = by_name[p]
        batch_holes = {round(normalize_angle(a), 6) for a in cfg["hole_angles"]}

        bad_add = [a for a in req.add_hole_angles if round(a, 6) not in batch_holes]
        if bad_add:
            raise MixedCorrectionConfigError(
                "加重孔不在批次配置的孔位中",
                {"plane": p, "unknown_hole_angles": bad_add,
                 "configured_hole_angles": cfg["hole_angles"]},
            )
        bad_drill = [
            h.hole_angle for h in req.drill_holes
            if round(normalize_angle(h.hole_angle), 6) not in batch_holes
        ]
        if bad_drill:
            raise MixedCorrectionConfigError(
                "钻削孔不在批次配置的孔位中",
                {"plane": p, "unknown_hole_angles": bad_drill},
            )

        occupied = {
            round(normalize_angle(w.hole_angle), 6)
            for w in req.existing_weights
            if round(normalize_angle(w.hole_angle), 6) in batch_holes
        }
        # 可用加重孔中被已有配重占据的孔自动剔除（同孔不可叠加），其余仍可用
        add_angles = [
            normalize_angle(a) for a in req.add_hole_angles
            if round(normalize_angle(a), 6) not in occupied
        ]

        existing_total = sum(w.mass for w in req.existing_weights)
        mass_limit = float(cfg["mass_limit"])
        if existing_total > mass_limit + 1e-9:
            raise MixedCorrectionConfigError(
                "已有配重总质量已超过该面质量上限",
                {"plane": p, "existing_mass": existing_total,
                 "mass_limit": mass_limit},
            )
        default_add = mass_limit - existing_total
        add_limit = req.add_mass_limit if req.add_mass_limit is not None else default_add
        if add_limit > default_add + 1e-9:
            raise MixedCorrectionConfigError(
                "加重质量上限超过该面剩余容量（面质量上限−已有配重）",
                {"plane": p, "requested_add_mass_limit": add_limit,
                 "available_capacity": default_add},
            )

        specs.append(
            mixed_core.build_plane_spec(
                add_hole_angles=add_angles,
                weight_specs=[float(m) for m in batch.weight_specs],
                drill_holes=[h.model_dump() for h in req.drill_holes],
                existing_weights=[w.model_dump() for w in req.existing_weights],
                add_mass_limit=add_limit,
                remove_mass_limit=req.remove_mass_limit,
                change_mass_limit=req.change_mass_limit,
                mass_per_mm=req.mass_per_mm,
                max_weights_per_plane=max_weights_per_plane,
            )
        )
    return specs


def _geometry_snapshot(planes: list[str], req_planes: list,
                       specs: list[mixed_core.PlaneSpec],
                       removal_step: float) -> dict:
    by_name = {p.plane: p for p in req_planes}
    out = {"planes": []}
    for p, spec in zip(planes, specs):
        req = by_name[p]
        out["planes"].append({
            "plane": p,
            "add_hole_angles": spec.add_hole_angles,
            "weight_specs": spec.weight_specs,
            "drill_holes": [
                {
                    "hole_angle": h.hole_angle,
                    "current_thickness": h.current_thickness,
                    "removal_limit": h.removal_limit,
                    "min_remaining_thickness": h.min_remaining_thickness,
                    "effective_removal_limit": spec.effective_hole_limit(h),
                }
                for h in spec.drill_holes
            ],
            "existing_weights": [
                {"hole_angle": w.hole_angle, "mass": w.mass}
                for w in spec.existing_weights
            ],
            "mass_per_mm": spec.mass_per_mm,
            "add_mass_limit": spec.add_mass_limit,
            "remove_mass_limit": spec.remove_mass_limit,
            "change_mass_limit": spec.change_mass_limit,
            "max_weights_per_plane": spec.max_weights_per_plane,
        })
    out["removal_step"] = removal_step
    out["snapshotted_at"] = datetime.now(timezone.utc).isoformat()
    return out


def _mixed_resolution_json(raw_v0, runout_vec, pred, sensors, amp_error,
                           phase_error) -> tuple[dict, list[str]]:
    k_err = amp_error + 2.0 * float(np.sin(np.deg2rad(phase_error) / 2.0))
    per_sensor, und = {}, []
    for i, s in enumerate(sensors):
        res = float((abs(raw_v0[i]) + abs(runout_vec[i])) * k_err)
        pred_amp = float(abs(pred[i]))
        flag = bool(pred_amp <= res)
        per_sensor[s] = {"resolution": res, "predicted_amplitude": pred_amp,
                         "undecidable": flag}
        if flag:
            und.append(s)
    return {"sensors": per_sensor, "undecidable_sensors": und}, und


def _candidate_to_json(
    c: mixed_core.MixedCandidate, planes: list[str], sensors: list[str],
    raw_v0, runout_vec, batch: Batch, rank: int,
) -> dict:
    actions_out = []
    for p, sel in enumerate(c.plane_selections):
        plane = planes[p]
        for a in sel.actions:
            ev = next(
                (e for e in c.evaluated_actions
                 if e.plane == plane and e.kind == a.kind
                 and round(e.hole_angle, 6) == round(a.hole_angle, 6)
                 and abs(e.mass - a.mass) < 1e-9),
                None,
            )
            actions_out.append({
                "plane": plane,
                "kind": a.kind,
                "hole_angle": a.hole_angle,
                "mass": float(a.mass),
                "drill_depth": ev.drill_depth if ev else None,
                "remaining_thickness": ev.remaining_thickness if ev else None,
                "thickness_margin": ev.thickness_margin if ev else None,
            })

    plane_weights = []
    for p, name in enumerate(planes):
        amp, phase = complex_to_amp_phase(c.weights[p])
        plane_actions = [a for a in actions_out if a["plane"] == name]
        plane_weights.append({
            "plane": name,
            "resultant_mass": amp,
            "angle": phase,
            "add_mass": float(sum(a["mass"] for a in plane_actions if a["kind"] == "add")),
            "remove_mass": float(sum(a["mass"] for a in plane_actions if a["kind"] == "remove")),
        })

    pred_out = {}
    for i, s in enumerate(sensors):
        amp, phase = complex_to_amp_phase(c.predicted_residual[i])
        pred_out[s] = {"amplitude": amp, "phase": phase}

    action_effects = []
    for e in c.evaluated_actions:
        action_effects.append({
            "plane": e.plane,
            "kind": e.kind,
            "hole_angle": e.hole_angle,
            "mass": float(e.mass),
            "vector": {"real": float(e.vector.real), "imag": float(e.vector.imag)},
            "drill_depth": e.drill_depth,
            "remaining_thickness": e.remaining_thickness,
            "thickness_margin": e.thickness_margin,
            "effects": {
                s: (lambda z: {"real": float(z.real), "imag": float(z.imag),
                               "amplitude": float(abs(z)),
                               "phase": float(np.rad2deg(np.angle(z)) % 360.0)})(
                    e.effects[s])
                for s in sensors
            },
        })

    res_json, _ = _mixed_resolution_json(
        raw_v0, runout_vec, c.predicted_residual, sensors,
        batch.amp_error, batch.phase_error,
    )
    return {
        "rank": rank,
        "actions": actions_out,
        "plane_weights": plane_weights,
        "predicted_residual": pred_out,
        "predicted_metric": c.predicted_metric,
        "total_change": c.total_change,
        "worst_case": c.worst_case,
        "min_safety_margin": c.min_safety_margin,
        "critical_constraint": c.critical_constraint,
        "plane_safety": c.plane_safety,
        "action_effects": action_effects,
        "resolution": res_json,
    }


def _mixed_baseline_compensation(batch, profile, baseline_run, raw_v0,
                                 runout_vec, sensors) -> dict | None:
    if profile is None:
        return None
    comps = compensate_vectors(
        raw_v0, runout_vec, sensors, batch.amp_error, batch.phase_error
    )
    dispersion_by = {s: profile.summary[s]["dispersion"] for s in sensors}
    comp_json, und = _compensation_json(comps, dispersion_by)
    return {
        **_profile_block(profile, baseline_run.phase_reference),
        "baseline_run_id": baseline_run.id,
        "sensors": comp_json,
        "undecidable_sensors": und,
    }


def search_mixed(db: Session, batch_id: int,
                 req: MixedSearchRequest) -> tuple[dict, list[mixed_core.PlaneSpec],
                                                   dict, float, Calibration,
                                                   RunoutProfile | None,
                                                   np.ndarray, np.ndarray, np.ndarray,
                                                   list[str], list[str], Run]:
    batch = get_batch_or_404(db, batch_id)
    cal, profile, raw_v0, v0, runout_vec, sensors, planes, baseline_run = _mixed_prepare(
        db, batch,
        calibration_id=req.calibration_id,
        runout_profile_id=req.runout_profile_id,
    )
    alpha = _coefficient_matrix(cal, sensors, planes)
    specs = _build_plane_specs(
        batch, planes, req.planes,
        max_weights_per_plane=req.max_weights_per_plane,
    )
    removal_step = (
        float(req.removal_step)
        if req.removal_step is not None
        else float(min(batch.weight_specs))
    )

    candidates = mixed_core.search_mixed_corrections(
        alpha=alpha,
        v0=v0,
        plane_specs=specs,
        plane_names=planes,
        sensor_names=sensors,
        weight_specs=[float(m) for m in batch.weight_specs],
        removal_step=removal_step,
        amp_error=batch.amp_error,
        phase_error_deg=batch.phase_error,
        angle_tolerance_deg=batch.angle_tolerance,
        baseline_amplitudes=np.abs(v0),
        max_weights_per_plane=req.max_weights_per_plane,
        keep_per_plane=req.keep_per_plane,
        top=req.top,
    )
    geometry = _geometry_snapshot(planes, req.planes, specs, removal_step)
    candidates_json = [
        _candidate_to_json(c, planes, sensors, raw_v0, runout_vec, batch, rank)
        for rank, c in enumerate(candidates, start=1)
    ]
    response = {
        "calibration_id": cal.id,
        "runout_profile_id": profile.id if profile else None,
        "phase_reference": baseline_run.phase_reference,
        "removal_step": removal_step,
        "geometry_snapshot": geometry,
        "candidates": candidates_json,
    }
    return (response, specs, geometry, removal_step, cal, profile, raw_v0, v0,
            runout_vec, sensors, planes, baseline_run)


def create_mixed_plan(
    db: Session, batch_id: int, req: MixedPlanCreate
) -> MixedPlan:
    batch = get_batch_or_404(db, batch_id)
    (_, specs, geometry, removal_step, cal, profile, raw_v0, v0, runout_vec,
     sensors, planes, baseline_run) = search_mixed(db, batch_id, req)
    alpha = _coefficient_matrix(cal, sensors, planes)

    # 按面切分确认动作并逐项复核（与搜索同一约束口径）
    actions_by_plane: dict[str, list[mixed_core.MixedAction]] = {p: [] for p in planes}
    unknown_planes = sorted({a.plane for a in req.actions if a.plane not in actions_by_plane})
    if unknown_planes:
        raise MixedCorrectionConfigError(
            "动作中的校正面与批次配置不符",
            {"unknown_planes": unknown_planes, "expected_planes": planes},
        )
    for a in req.actions:
        actions_by_plane[a.plane].append(
            mixed_core.MixedAction(
                kind=a.kind,
                hole_angle=normalize_angle(a.hole_angle),
                mass=float(a.mass),
            )
        )

    selections: list[mixed_core.PlaneSelection] = []
    for p, name in enumerate(planes):
        selections.append(
            mixed_core.assemble_plane_actions(
                specs[p], actions_by_plane[name], removal_step, name
            )
        )

    c = mixed_core.evaluate_mixed(
        alpha=alpha,
        v0=v0,
        plane_selections=selections,
        plane_names=planes,
        plane_specs=specs,
        sensor_names=sensors,
        amp_error=batch.amp_error,
        phase_error_deg=batch.phase_error,
        angle_tolerance_deg=batch.angle_tolerance,
        baseline_amplitudes=np.abs(v0),
    )
    c_json = _candidate_to_json(
        c, planes, sensors, raw_v0, runout_vec, batch, req.candidate_rank or 1
    )
    baseline_comp = _mixed_baseline_compensation(
        batch, profile, baseline_run, raw_v0, runout_vec, sensors
    )
    constraints = {
        "calibration_id": cal.id,
        "amp_error": batch.amp_error,
        "phase_error": batch.phase_error,
        "angle_tolerance": batch.angle_tolerance,
        "removal_step": removal_step,
        "candidate_rank": req.candidate_rank,
        "reference_speed": batch.reference_speed,
        "phase_reference": baseline_run.phase_reference,
    }

    plan = MixedPlan(
        batch_id=batch.id,
        calibration_id=cal.id,
        runout_profile_id=profile.id if profile else None,
        name=req.name,
        actions=c_json["actions"],
        predicted_residual=c_json["predicted_residual"],
        predicted_metric=c.predicted_metric,
        total_change=c.total_change,
        worst_case=c.worst_case,
        min_safety_margin=c.min_safety_margin,
        critical_constraint=c.critical_constraint,
        plane_safety=c_json["plane_safety"],
        action_effects=c_json["action_effects"],
        resolution=c_json["resolution"],
        geometry_snapshot=geometry,
        constraints_snapshot=constraints,
        runout_compensation=baseline_comp,
        note="",
    )
    db.add(plan)
    db.commit()
    db.refresh(plan)
    return plan


def plan_out(plan: MixedPlan) -> dict:
    return {
        "id": plan.id,
        "batch_id": plan.batch_id,
        "calibration_id": plan.calibration_id,
        "runout_profile_id": plan.runout_profile_id,
        "name": plan.name,
        "actions": plan.actions,
        "predicted_residual": plan.predicted_residual,
        "predicted_metric": plan.predicted_metric,
        "total_change": plan.total_change,
        "worst_case": plan.worst_case,
        "min_safety_margin": plan.min_safety_margin,
        "critical_constraint": plan.critical_constraint,
        "plane_safety": plan.plane_safety,
        "action_effects": plan.action_effects,
        "resolution": plan.resolution,
        "geometry_snapshot": plan.geometry_snapshot,
        "constraints_snapshot": plan.constraints_snapshot,
        "runout_compensation": plan.runout_compensation,
        "note": plan.note,
        "created_at": plan.created_at,
        "verification_count": len(plan.verifications),
    }


# ---------------------------------------------------------------- 混合校正复测


def _sensor_comparison_entry(
    *, sensor: str, raw_z, net_z, runout_z, z_pred, base_amp,
    amp_error: float, phase_error: float,
) -> dict:
    """构造逐测点“预测 vs 实测（净）”对比条目，普通复测与混合复测共用。"""
    r_amp, r_phase = complex_to_amp_phase(raw_z)
    c_amp, c_phase = complex_to_amp_phase(runout_z)
    n_amp, n_phase = complex_to_amp_phase(net_z)
    resolution = (abs(raw_z) + abs(runout_z)) * (
        amp_error + 2.0 * float(np.sin(np.deg2rad(phase_error) / 2.0))
    )
    undecidable = bool(n_amp <= resolution)
    dev = abs(net_z - z_pred)
    entry = {
        "predicted": (lambda z: {"amplitude": abs(z),
                                 "phase": float(np.rad2deg(np.angle(z)) % 360.0)})(z_pred),
        "raw_measured": {"amplitude": r_amp, "phase": r_phase},
        "runout_compensation": {"amplitude": c_amp, "phase": c_phase},
        "measured": {"amplitude": n_amp, "phase": n_phase},
        "resolution": float(resolution),
        "undecidable": undecidable,
        "verdict": "undecidable" if undecidable else "decidable",
        "vector_deviation": float(dev),
        "relative_deviation": float(dev / max(abs(net_z), 1e-12)),
    }
    if base_amp is not None and base_amp > 0:
        entry["baseline_amplitude"] = base_amp
        entry["reduction_ratio"] = (
            None if undecidable else float(1.0 - n_amp / base_amp)
        )
    return entry


def add_mixed_verification(
    db: Session, plan_id: int, payload: MixedVerificationCreate
) -> MixedVerification:
    plan = db.get(MixedPlan, plan_id)
    if plan is None:
        raise NotFoundError(f"混合校正方案 {plan_id} 不存在", {"plan_id": plan_id})
    batch = plan.batch
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
            {"reference_speed": batch.reference_speed,
             "speed_tolerance": batch.speed_tolerance,
             "runs": [{"speed": payload.speed,
                       "deviation": abs(payload.speed - batch.reference_speed)}]},
        )

    baselines = [r for r in batch.runs if r.kind == "baseline"]
    baseline_run = baselines[0] if baselines else None
    expected_ref = baseline_run.phase_reference if baseline_run else None
    phase_ref = payload.phase_reference or expected_ref
    if expected_ref is not None and phase_ref != expected_ref:
        raise PhaseReferenceConflict(
            "复测相位基准与批次运行不一致",
            {"references": {
                expected_ref: [baseline_run.id] if baseline_run else [],
                phase_ref: ["verification"],
            }},
        )

    profile_id = (
        payload.runout_profile_id
        if payload.runout_profile_id is not None
        else plan.runout_profile_id
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
                {"runout_profile_id": profile.id,
                 "runout_phase_references": sorted(record_refs),
                 "verification_phase_reference": phase_ref,
                 "record_ids": [r.id for r in profile.records]},
            )
        runout_vec = _runout_mean_vector(profile, sensors)

    # ---- 逐项核对实际加重/去料
    planned = [
        {**a, "_angle": round(normalize_angle(a["hole_angle"]), 6)}
        for a in plan.actions
    ]
    planes = _plane_names(batch)
    cal = db.get(Calibration, plan.calibration_id)
    alpha = _coefficient_matrix(cal, sensors, planes) if cal is not None else None

    def _match_plan(plane: str, kind: str, ang: float):
        for i, p in enumerate(planned):
            if (p["plane"] == plane and p["kind"] == kind
                    and p["_angle"] == round(ang, 6)):
                return i, p
        return None, None

    actual_actions_out: list[dict] = []
    unmatched_actual: list[dict] = []
    matched_idx: set[int] = set()
    delta_w = {p: 0j for p in planes}

    for aa in payload.actual_actions:
        ang = normalize_angle(aa.hole_angle)
        idx, p = _match_plan(aa.plane, aa.kind, ang)
        if p is None:
            unmatched_actual.append(
                {"plane": aa.plane, "kind": aa.kind, "hole_angle": ang,
                 "mass": aa.mass, "reason": "no_matching_planned_action"}
            )
            continue
        matched_idx.add(idx)
        sign = 1.0 if aa.kind == "add" else -1.0
        vec_actual = sign * (aa.mass if aa.executed else 0.0) * np.exp(
            1j * np.deg2rad(ang))
        vec_planned = sign * p["mass"] * np.exp(1j * np.deg2rad(ang))
        dvec = vec_actual - vec_planned
        delta_w[aa.plane] += dvec

        # 去料厚度核对
        thickness = None
        geo_plane = next(
            g for g in plan.geometry_snapshot["planes"] if g["plane"] == aa.plane
        )
        if aa.kind == "remove":
            mass_per_mm = geo_plane["mass_per_mm"]
            hole = next(
                (h for h in geo_plane["drill_holes"]
                 if round(h["hole_angle"], 6) == round(ang, 6)), None,
            )
            if hole is not None and mass_per_mm:
                depth = (aa.mass if aa.executed else 0.0) / mass_per_mm
                remaining = hole["current_thickness"] - depth
                margin = remaining - hole["min_remaining_thickness"]
                thickness = {
                    "drill_depth": depth,
                    "remaining_thickness": remaining,
                    "min_remaining_thickness": hole["min_remaining_thickness"],
                    "thickness_margin": margin,
                    "violated": bool(margin < -1e-9),
                }

            status = "ok" if aa.executed and abs(aa.mass - p["mass"]) <= 1e-7 \
                else ("not_executed" if not aa.executed else "mass_deviation")
            if thickness is not None and thickness["violated"]:
                status = "thickness_violated" if status == "ok" else \
                    status + "+thickness_violated"
            actual_actions_out.append({
                "plane": aa.plane,
                "kind": aa.kind,
                "hole_angle": ang,
                "planned_mass": p["mass"],
                "actual_mass": aa.mass if aa.executed else 0.0,
                "executed": aa.executed,
                "mass_deviation": float((aa.mass if aa.executed else 0.0) - p["mass"]),
                "vector_deviation": {"real": float(dvec.real), "imag": float(dvec.imag),
                                     "magnitude": float(abs(dvec))},
                "thickness": thickness,
                "status": status,
            })

    missing = [
        {"plane": p["plane"], "kind": p["kind"], "hole_angle": p["hole_angle"],
         "planned_mass": p["mass"]}
        for i, p in enumerate(planned) if i not in matched_idx
    ]

    action_verdict = "match"
    if unmatched_actual or missing:
        action_verdict = "mismatch"
    elif any(a["status"] != "ok" for a in actual_actions_out):
        action_verdict = "deviation"
    # ---- 残振对比
    by_sensor = {m.sensor: m for m in payload.measurements}
    raw_vec = np.array([
        amp_phase_to_complex(by_sensor[s].amplitude, by_sensor[s].phase)
        for s in sensors
    ])

    # 基线净振幅取自方案的轴跳快照
    baseline_net_amp: dict[str, float] = {}
    if baseline_run is not None:
        raw_base = {m["sensor"]: m for m in baseline_run.measurements}
        block = plan.runout_compensation
        for s in sensors:
            if block and s in block["sensors"]:
                baseline_net_amp[s] = block["sensors"][s]["net"]["amplitude"]
            else:
                baseline_net_amp[s] = raw_base[s]["amplitude"]

    comparison: dict = {"sensors": {}, "undecidable_sensors": []}
    undecidable_all: list[str] = []
    if profile is not None:
        comparison["runout_profile"] = _profile_block(profile, phase_ref)

    # 方案预测 + 实际动作偏差经影响系数折算的“按实际动作预测”
    delta_contrib = None
    if alpha is not None:
        delta_contrib = alpha @ np.array([delta_w[p] for p in planes])

    for i, s in enumerate(sensors):
        pred = plan.predicted_residual[s]
        z_pred = amp_phase_to_complex(pred["amplitude"], pred["phase"])
        net = raw_vec[i] - runout_vec[i]
        z_pred_actual = (
            z_pred + delta_contrib[i] if delta_contrib is not None else z_pred
        )
        entry = _sensor_comparison_entry(
            sensor=s, raw_z=raw_vec[i], net_z=net, runout_z=runout_vec[i],
            z_pred=z_pred, base_amp=baseline_net_amp.get(s),
            amp_error=batch.amp_error, phase_error=batch.phase_error,
        )
        entry["predicted_with_actual_actions"] = {
            "amplitude": float(abs(z_pred_actual)),
            "phase": float(np.rad2deg(np.angle(z_pred_actual)) % 360.0),
            "real": float(z_pred_actual.real),
            "imag": float(z_pred_actual.imag),
        }
        entry["deviation_from_actual_action_prediction"] = float(
            abs(net - z_pred_actual)
        )
        if entry["undecidable"]:
            undecidable_all.append(s)
        comparison["sensors"][s] = entry

    comparison["undecidable_sensors"] = undecidable_all
    comparison["balance_verdict"] = (
        "undecidable" if undecidable_all else "decidable"
    )
    comparison["action_check_verdict"] = action_verdict
    comparison["missing_actions"] = missing
    comparison["unmatched_actual_actions"] = unmatched_actual
    comparison["plane_vector_deviation"] = {
        p: {"real": float(delta_w[p].real), "imag": float(delta_w[p].imag),
            "magnitude": float(abs(delta_w[p]))}
        for p in planes
    }

    ver = MixedVerification(
        plan_id=plan.id,
        runout_profile_id=profile.id if profile else None,
        speed=payload.speed,
        measurements=[m.model_dump() for m in payload.measurements],
        actual_actions=actual_actions_out,
        reconciliation=comparison,
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
    for mp in batch.mixed_plans:
        if mp.runout_profile_id and str(mp.runout_profile_id) in usage:
            usage[str(mp.runout_profile_id)].setdefault(
                "mixed_plan_ids", []
            ).append(mp.id)
        for mv in mp.verifications:
            if mv.runout_profile_id and str(mv.runout_profile_id) in usage:
                usage[str(mv.runout_profile_id)].setdefault(
                    "mixed_verification_ids", []
                ).append(mv.id)
    for u in usage.values():
        u.setdefault("mixed_plan_ids", [])
        u.setdefault("mixed_verification_ids", [])

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
        "mixed_plans": [
            {
                "id": mp.id,
                "calibration_id": mp.calibration_id,
                "runout_profile_id": mp.runout_profile_id,
                "name": mp.name,
                "actions": mp.actions,
                "predicted_residual": mp.predicted_residual,
                "predicted_metric": mp.predicted_metric,
                "total_change": mp.total_change,
                "worst_case": mp.worst_case,
                "min_safety_margin": mp.min_safety_margin,
                "critical_constraint": mp.critical_constraint,
                "plane_safety": mp.plane_safety,
                "action_effects": mp.action_effects,
                "resolution": mp.resolution,
                "geometry_snapshot": mp.geometry_snapshot,
                "constraints_snapshot": mp.constraints_snapshot,
                "runout_compensation": mp.runout_compensation,
                "note": mp.note,
                "created_at": mp.created_at.isoformat(),
                "verifications": [
                    {
                        "id": mv.id,
                        "runout_profile_id": mv.runout_profile_id,
                        "speed": mv.speed,
                        "measurements": mv.measurements,
                        "actual_actions": mv.actual_actions,
                        "reconciliation": mv.reconciliation,
                        "note": mv.note,
                        "created_at": mv.created_at.isoformat(),
                    }
                    for mv in mp.verifications
                ],
            }
            for mp in batch.mixed_plans
        ],
    }
