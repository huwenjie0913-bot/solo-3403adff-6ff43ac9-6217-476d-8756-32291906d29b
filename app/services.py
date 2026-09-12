"""业务编排：校验、标定、求解、复测、导出。"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sqlalchemy.orm import Session

from .core.correction import solve_continuous
from .core.discrete import search_discrete_solutions, worst_case_residual
from .core.influence import fit_influence_coefficients
from .core.vibration import amp_phase_to_complex, complex_to_amp_phase
from .errors import (
    CalibrationMissing,
    DomainError,
    InsufficientTrials,
    NotFoundError,
    PhaseReferenceConflict,
    SpeedDeviationError,
)
from .models import Batch, Calibration, Run, Solution, Verification
from .schemas import BatchCreate, RunCreate, SolveRequest, VerificationCreate


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


# ---------------------------------------------------------------- 标定


def calibrate(db: Session, batch_id: int) -> Calibration:
    batch = get_batch_or_404(db, batch_id)
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
    baseline_vec = _run_vector(baselines[0], sensors)

    trial_matrix = np.zeros((len(trials), len(planes)), dtype=complex)
    for k, run in enumerate(trials):
        for tw in run.trial_weights:
            p = planes.index(tw["plane"])
            trial_matrix[k, p] += amp_phase_to_complex(tw["mass"], tw["angle"])
    trial_vectors = np.array([_run_vector(r, sensors) for r in trials])

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
        "calibrated_at": datetime.now(timezone.utc).isoformat(),
    }

    if batch.calibration is not None:
        db.delete(batch.calibration)
        db.flush()
    cal = Calibration(
        batch_id=batch.id,
        coefficients=coefficients,
        residuals=residuals,
        condition=result.condition,
        provenance=provenance,
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
    cal = batch.calibration
    if cal is None:
        cal = calibrate(db, batch_id)  # 未标定时自动标定
    if cal is None:  # pragma: no cover - 防御
        raise CalibrationMissing("批次尚未标定")

    sensors = _sensor_names(batch)
    planes = _plane_names(batch)
    alpha = _coefficient_matrix(cal, sensors, planes)

    baselines = [r for r in batch.runs if r.kind == "baseline"]
    v0 = _run_vector(baselines[0], sensors)

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

    # 旧的未复测方案视为被取代，删除；已复测的保留为历史记录
    for old in list(batch.solutions):
        if not old.verifications:
            db.delete(old)
    db.flush()

    def _residual_json(pred: np.ndarray) -> dict:
        out = {}
        for i, s in enumerate(sensors):
            amp, phase = complex_to_amp_phase(pred[i])
            out[s] = {"amplitude": amp, "phase": phase}
        return out

    cont_sol = Solution(
        batch_id=batch.id,
        kind="continuous",
        rank=0,
        weights=[
            {
                "plane": planes[p],
                "mass": float(abs(cont.weights[p])),
                "angle": float(np.rad2deg(np.angle(cont.weights[p])) % 360.0),
                "assignments": [],
            }
            for p in range(len(planes))
        ],
        predicted_residual=_residual_json(cont.predicted_residual),
        predicted_metric=float(np.max(np.abs(cont.predicted_residual))),
        total_mass=float(np.sum(np.abs(cont.weights))),
        # 连续解按“精确配重以安装角公差安装”评估同一最差情形界
        worst_case=worst_case_residual(
            cont.predicted_residual,
            alpha,
            [float(abs(w)) for w in cont.weights],
            np.abs(v0),
            batch.amp_error,
            batch.phase_error,
            batch.angle_tolerance,
        ),
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
        sol = Solution(
            batch_id=batch.id,
            kind="discrete",
            rank=rank,
            weights=weights,
            predicted_residual=_residual_json(d.predicted_residual),
            predicted_metric=d.predicted_metric,
            total_mass=d.total_mass,
            worst_case=d.worst_case,
        )
        db.add(sol)
        discrete_sols.append(sol)
    db.commit()
    db.refresh(cont_sol)
    for s in discrete_sols:
        db.refresh(s)
    return cont_sol, discrete_sols


# ---------------------------------------------------------------- 复测


def add_verification(db: Session, solution_id: int, payload: VerificationCreate) -> Verification:
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
    baseline_by_sensor = {}
    if baselines:
        baseline_by_sensor = {m["sensor"]: m for m in baselines[0].measurements}

    comparison: dict = {"sensors": {}, "max_relative_deviation": 0.0}
    for m in payload.measurements:
        pred = sol.predicted_residual[m.sensor]
        z_pred = amp_phase_to_complex(pred["amplitude"], pred["phase"])
        z_meas = amp_phase_to_complex(m.amplitude, m.phase)
        dev = abs(z_meas - z_pred)
        rel = float(dev / max(abs(z_meas), 1e-12))
        entry = {
            "predicted": {"amplitude": pred["amplitude"], "phase": pred["phase"]},
            "measured": {"amplitude": m.amplitude, "phase": m.phase},
            "vector_deviation": float(dev),
            "relative_deviation": rel,
        }
        base = baseline_by_sensor.get(m.sensor)
        if base and base["amplitude"] > 0:
            entry["baseline_amplitude"] = base["amplitude"]
            entry["reduction_ratio"] = float(1.0 - m.amplitude / base["amplitude"])
        comparison["sensors"][m.sensor] = entry
        comparison["max_relative_deviation"] = max(comparison["max_relative_deviation"], rel)

    ver = Verification(
        solution_id=sol.id,
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


def export_batch(db: Session, batch_id: int) -> dict:
    batch = get_batch_or_404(db, batch_id)
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
        "calibration": (
            {
                "coefficients": batch.calibration.coefficients,
                "residuals": batch.calibration.residuals,
                "condition": batch.calibration.condition,
                "provenance": batch.calibration.provenance,
                "created_at": batch.calibration.created_at.isoformat(),
            }
            if batch.calibration
            else None
        ),
        "solutions": [
            {
                "id": s.id,
                "kind": s.kind,
                "rank": s.rank,
                "weights": s.weights,
                "predicted_residual": s.predicted_residual,
                "predicted_metric": s.predicted_metric,
                "total_mass": s.total_mass,
                "worst_case": s.worst_case,
                "created_at": s.created_at.isoformat(),
                "verifications": [
                    {
                        "id": v.id,
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
