"""加重/去料混合校正：联合枚举、方案快照、复测逐项核对与冲突诊断。"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.vibration import complex_to_amp_phase
from app.main import app

ALPHA_TRUE = np.array(
    [
        [0.08 * np.exp(1j * np.deg2rad(30)), 0.05 * np.exp(1j * np.deg2rad(120))],
        [0.06 * np.exp(1j * np.deg2rad(-60)), 0.09 * np.exp(1j * np.deg2rad(200))],
    ]
)
UNBALANCE = np.array(
    [12 * np.exp(1j * np.deg2rad(45)), 8 * np.exp(1j * np.deg2rad(250))]
)
REF_SPEED = 3000.0
T1 = np.array([20 * np.exp(1j * np.deg2rad(0)), 0])
T2 = np.array([0, 20 * np.exp(1j * np.deg2rad(90))])
V0 = ALPHA_TRUE @ UNBALANCE
SENSORS = ["DE", "NDE"]
HOLES = [i * 30 for i in range(12)]


def meas(v: np.ndarray) -> list[dict]:
    return [
        {"sensor": name, "amplitude": amp, "phase": phase}
        for name, (amp, phase) in zip(SENSORS, (complex_to_amp_phase(z) for z in v))
    ]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def make_batch(client, **overrides) -> int:
    payload = {
        "name": "转子-混合校正",
        "reference_speed": REF_SPEED,
        "speed_tolerance": 30.0,
        "planes": [
            {"name": "P1", "correction_radius": 250.0,
             "hole_angles": HOLES, "mass_limit": 100.0},
            {"name": "P2", "correction_radius": 250.0,
             "hole_angles": HOLES, "mass_limit": 100.0},
        ],
        "sensors": [{"name": "DE"}, {"name": "NDE"}],
        "weight_specs": [2, 5, 10, 20],
        "amp_error": 0.02,
        "phase_error": 2.0,
        "angle_tolerance": 5.0,
    }
    payload.update(overrides)
    r = client.post("/api/batches", json=payload)
    assert r.status_code == 201, r.text
    return r.json()["id"]


def balanced(client) -> int:
    bid = make_batch(client)
    client.post(f"/api/batches/{bid}/runs", json={
        "kind": "baseline", "speed": REF_SPEED, "phase_reference": "lag",
        "measurements": meas(V0)})
    client.post(f"/api/batches/{bid}/runs", json={
        "kind": "trial", "speed": REF_SPEED, "phase_reference": "lag",
        "measurements": meas(V0 + ALPHA_TRUE @ T1),
        "trial_weights": [{"plane": "P1", "mass": 20, "angle": 0}]})
    client.post(f"/api/batches/{bid}/runs", json={
        "kind": "trial", "speed": REF_SPEED, "phase_reference": "lag",
        "measurements": meas(V0 + ALPHA_TRUE @ T2),
        "trial_weights": [{"plane": "P2", "mass": 20, "angle": 90}]})
    cal = client.post(f"/api/batches/{bid}/calibrate")
    assert cal.status_code == 200, cal.text
    return bid


def mixed_planes(**overrides) -> list[dict]:
    """默认两面都可加可钻、无已有配重的逐面几何录入。"""
    def one(plane):
        return {
            "plane": plane,
            "add_hole_angles": HOLES,
            "drill_holes": [
                {"hole_angle": a, "current_thickness": 10.0,
                 "removal_limit": 20.0, "min_remaining_thickness": 2.0}
                for a in HOLES
            ],
            "existing_weights": [],
            "mass_per_mm": 0.5,
        }

    planes = [one("P1"), one("P2")]
    out = {p["plane"]: p for p in planes}
    for name, patch in overrides.items():
        out[name].update(patch)
    return [out["P1"], out["P2"]]


SEARCH_BODY = {"planes": None, "removal_step": 2.0, "top": 8}


def search(client, bid, planes, **kw):
    body = {**SEARCH_BODY, "planes": planes, **kw}
    return client.post(f"/api/batches/{bid}/mixed-corrections/search", json=body)


# ------------------------------------------------------------ 搜索


def test_search_returns_ranked_mixed_candidates(client):
    bid = balanced(client)
    r = search(client, bid, mixed_planes())
    assert r.status_code == 200, r.text
    body = r.json()
    cands = body["candidates"]
    assert len(cands) == 8
    assert body["removal_step"] == 2.0
    assert body["phase_reference"] == "lag"
    assert body["runout_profile_id"] is None

    metrics = [c["predicted_metric"] for c in cands]
    assert metrics == sorted(metrics)
    # 最优候选应显著降低振动（基线约 1 mm/s 量级）
    assert metrics[0] < 0.2 * float(np.max(np.abs(V0)))

    best = cands[0]
    assert best["rank"] == 1
    assert best["worst_case"] >= best["predicted_metric"]
    for pw in best["plane_weights"]:
        assert pw["add_mass"] + pw["remove_mass"] <= 100.0 + 1e-9
    # 几何快照包含钻削孔有效上限（min(每孔上限, 厚度换算 4g)）
    geo = body["geometry_snapshot"]["planes"]
    assert geo[0]["plane"] == "P1"
    assert geo[0]["drill_holes"][0]["effective_removal_limit"] == pytest.approx(4.0)
    # 默认改变量上限 = 面质量上限(100) + 各孔去料有效容量之和(12×4=48)
    assert geo[0]["change_mass_limit"] == pytest.approx(148.0)

    # 安全余量：所有面的加/去/改/厚度余量非负
    for safety in best["plane_safety"]:
        assert safety["add_headroom"] >= -1e-9
        assert safety["remove_headroom"] >= -1e-9
        assert safety["change_headroom"] >= -1e-9
        for h in safety["drill_holes"]:
            assert h["thickness_margin"] >= -1e-9
            assert h["remaining_thickness"] >= h["min_remaining_thickness"] - 1e-9
    assert 0.0 <= best["min_safety_margin"] <= 1.0

    # 逐动作对测点的预测贡献：加重 +、去料 −，按 α·w 计算
    assert best["action_effects"]
    eff0 = best["action_effects"][0]
    assert set(eff0["effects"]) == {"DE", "NDE"}


def test_removal_is_reverse_vector_and_reduces_vibration(client):
    bid = balanced(client)
    r = search(client, bid, mixed_planes())
    best = r.json()["candidates"][0]

    # 用候选动作重建合成向量：去料必须取负号
    w = np.zeros(2, dtype=complex)
    plane_idx = {"P1": 0, "P2": 1}
    for a in best["actions"]:
        sign = 1.0 if a["kind"] == "add" else -1.0
        w[plane_idx[a["plane"]]] += sign * a["mass"] * np.exp(
            1j * np.deg2rad(a["hole_angle"]))
    pred = V0 + ALPHA_TRUE @ w
    for i, s in enumerate(SENSORS):
        assert abs(pred[i]) == pytest.approx(
            best["predicted_residual"][s]["amplitude"], abs=1e-6)

    # 若去料错误取正号，结果必然不同（验证反向约定确实参与求解）
    w_wrong = np.zeros(2, dtype=complex)
    for a in best["actions"]:
        w_wrong[plane_idx[a["plane"]]] += a["mass"] * np.exp(
            1j * np.deg2rad(a["hole_angle"]))
    if any(a["kind"] == "remove" for a in best["actions"]):
        assert abs(np.max(np.abs(V0 + ALPHA_TRUE @ w_wrong)) -
                   best["predicted_metric"]) > 1e-6


def test_sort_uses_total_change_and_safety_margin(client):
    bid = balanced(client)
    r = search(client, bid, mixed_planes(), top=10)
    cands = r.json()["candidates"]
    keys = [
        (round(c["predicted_metric"], 6), round(c["total_change"], 6),
         round(-c["min_safety_margin"], 6))
        for c in cands
    ]
    assert keys == sorted(keys)


def test_existing_weights_occupy_hole_and_capacity(client):
    bid = balanced(client)
    planes = mixed_planes(P1={
        "existing_weights": [{"hole_angle": 0.0, "mass": 95.0}],
    })
    r = search(client, bid, planes)
    assert r.status_code == 200, r.text
    geo = r.json()["geometry_snapshot"]["planes"][0]
    # 加重余量 = 100 − 95 = 5g；0° 孔被占据
    assert geo["add_mass_limit"] == pytest.approx(5.0)
    for cand in r.json()["candidates"]:
        assert all(
            not (a["plane"] == "P1" and a["kind"] == "add" and a["hole_angle"] == 0.0)
            for a in cand["actions"])
        for pw in cand["plane_weights"]:
            if pw["plane"] == "P1":
                assert pw["add_mass"] <= 5.0 + 1e-9


# ------------------------------------------------------------ 无解/配置冲突


def test_no_feasible_points_to_planes_and_constraints(client):
    bid = balanced(client)
    # P1 不能加重（上限 0）也不能钻（无孔）；P2 正常
    planes = mixed_planes(P1={
        "add_hole_angles": [],
        "drill_holes": [],
        "add_mass_limit": 0.0,
    })
    r = search(client, bid, planes)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "no_feasible_mixed_correction"
    p1 = next(b for b in body["details"]["planes"] if b["plane"] == "P1")
    codes = {b["code"] for b in body["details"]["planes"] if b["plane"] == "P1"}
    assert {"no_add_holes", "no_drill_holes", "add_mass_limit_saturated"} <= codes
    assert "required_resultant_mass" in body["details"]


def test_config_error_existing_weight_over_capacity(client):
    bid = balanced(client)
    planes = mixed_planes(P1={
        "existing_weights": [{"hole_angle": 0.0, "mass": 150.0}],
    })
    r = search(client, bid, planes)
    assert r.status_code == 422
    assert r.json()["error"] == "mixed_correction_config_error"


def test_config_error_thickness_infeasible_hole(client):
    bid = balanced(client)
    planes = mixed_planes(P1={
        "drill_holes": [
            {"hole_angle": 30.0, "current_thickness": 1.0,
             "removal_limit": 10.0, "min_remaining_thickness": 2.0}
        ],
    })
    r = search(client, bid, planes)
    assert r.status_code == 422
    assert r.json()["error"] == "mixed_correction_config_error"


def test_config_error_mass_per_mm_required_for_drilling(client):
    bid = balanced(client)
    planes = mixed_planes(P1={"mass_per_mm": None})
    r = search(client, bid, planes)
    assert r.status_code == 422
    assert r.json()["error"] == "mixed_correction_config_error"


def test_config_error_unknown_plane(client):
    bid = balanced(client)
    planes = mixed_planes()
    planes[0]["plane"] = "PX"
    r = search(client, bid, planes)
    assert r.status_code == 422
    assert r.json()["error"] == "mixed_correction_config_error"
    assert "PX" in r.json()["details"]["unknown_planes"]


def test_search_requires_existing_calibration(client):
    bid = make_batch(client)
    r = search(client, bid, mixed_planes())
    assert r.status_code == 422
    assert r.json()["error"] == "calibration_missing"


# ------------------------------------------------------------ 确认保存


def test_confirm_persists_plan_with_snapshot(client):
    bid = balanced(client)
    sr = search(client, bid, mixed_planes())
    best = sr.json()["candidates"][0]
    actions = [
        {"plane": a["plane"], "kind": a["kind"],
         "hole_angle": a["hole_angle"], "mass": a["mass"]}
        for a in best["actions"]
    ]
    r = client.post(f"/api/batches/{bid}/mixed-corrections", json={
        "planes": sr.json()["geometry_snapshot"] and _planes_from_search(sr.json()),
        "removal_step": 2.0,
        "actions": actions,
        "candidate_rank": 1,
        "name": "方案-混合-01",
    })
    assert r.status_code == 201, r.text
    plan = r.json()
    assert plan["name"] == "方案-混合-01"
    assert plan["calibration_id"]
    assert plan["predicted_metric"] == pytest.approx(best["predicted_metric"])
    assert plan["total_change"] == pytest.approx(best["total_change"])
    assert plan["min_safety_margin"] == pytest.approx(best["min_safety_margin"])
    # 几何与约束快照
    assert plan["geometry_snapshot"]["planes"][0]["plane"] == "P1"
    assert plan["constraints_snapshot"]["removal_step"] == 2.0
    assert plan["constraints_snapshot"]["calibration_id"] == plan["calibration_id"]
    # 动作带面名与去料深度
    removes = [a for a in plan["actions"] if a["kind"] == "remove"]
    if removes:
        assert removes[0]["drill_depth"] == pytest.approx(removes[0]["mass"] / 0.5)

    listing = client.get(f"/api/batches/{bid}/mixed-corrections").json()
    assert any(p["id"] == plan["id"] for p in listing)
    got = client.get(f"/api/mixed-corrections/{plan['id']}").json()
    assert got["id"] == plan["id"]
    return plan, bid


def _planes_from_search(search_body: dict) -> list[dict]:
    """从搜索响应的几何快照重建确认请求的 planes 字段。"""
    out = []
    for g in search_body["geometry_snapshot"]["planes"]:
        out.append({
            "plane": g["plane"],
            "add_hole_angles": g["add_hole_angles"],
            "drill_holes": [
                {"hole_angle": h["hole_angle"],
                 "current_thickness": h["current_thickness"],
                 "removal_limit": h["removal_limit"],
                 "min_remaining_thickness": h["min_remaining_thickness"]}
                for h in g["drill_holes"]
            ],
            "existing_weights": g["existing_weights"],
            "mass_per_mm": g["mass_per_mm"],
            "add_mass_limit": g["add_mass_limit"],
            "remove_mass_limit": g["remove_mass_limit"],
            "change_mass_limit": g["change_mass_limit"],
        })
    return out


def test_confirm_rejects_hole_and_thickness_violation(client):
    bid = balanced(client)
    sr = search(client, bid, mixed_planes())
    planes = _planes_from_search(sr.json())
    # 在非加重孔（15°）加重 -> 拒绝
    r = client.post(f"/api/batches/{bid}/mixed-corrections", json={
        "planes": planes, "removal_step": 2.0,
        "actions": [{"plane": "P1", "kind": "add", "hole_angle": 15.0, "mass": 10}],
    })
    assert r.status_code == 422
    assert r.json()["error"] == "mixed_correction_config_error"

    # 去料超过厚度有效上限（10mm 厚、剩 2mm、0.5g/mm -> 至多 4g）
    r = client.post(f"/api/batches/{bid}/mixed-corrections", json={
        "planes": planes, "removal_step": 2.0,
        "actions": [{"plane": "P1", "kind": "remove",
                     "hole_angle": 30.0, "mass": 10}],
    })
    assert r.status_code == 422
    assert r.json()["error"] == "mixed_correction_config_error"

    # 去料质量不是步长整数倍
    r = client.post(f"/api/batches/{bid}/mixed-corrections", json={
        "planes": planes, "removal_step": 2.0,
        "actions": [{"plane": "P1", "kind": "remove",
                     "hole_angle": 30.0, "mass": 3.0}],
    })
    assert r.status_code == 422
    assert r.json()["error"] == "mixed_correction_config_error"


def test_confirm_empty_plan_allowed(client):
    bid = balanced(client)
    sr = search(client, bid, mixed_planes())
    planes = _planes_from_search(sr.json())
    r = client.post(f"/api/batches/{bid}/mixed-corrections", json={
        "planes": planes, "removal_step": 2.0, "actions": [], "name": "不动作",
    })
    assert r.status_code == 201, r.text
    plan = r.json()
    assert plan["total_change"] == 0.0
    for s in SENSORS:
        assert plan["predicted_residual"][s]["amplitude"] == pytest.approx(
            abs(V0[SENSORS.index(s)]))


# ------------------------------------------------------------ 复测逐项核对


def _confirm_first_candidate(client, bid, planes_body, **search_kw):
    sr = search(client, bid, planes_body, **search_kw)
    assert sr.status_code == 200, sr.text
    body = sr.json()
    best = body["candidates"][0]
    actions = [
        {"plane": a["plane"], "kind": a["kind"],
         "hole_angle": a["hole_angle"], "mass": a["mass"]}
        for a in best["actions"]
    ]
    confirm_body = {
        "planes": _planes_from_search(body),
        "removal_step": body["removal_step"],
        "actions": actions,
    }
    if "runout_profile_id" in search_kw:
        confirm_body["runout_profile_id"] = search_kw["runout_profile_id"]
    r = client.post(f"/api/batches/{bid}/mixed-corrections", json=confirm_body)
    assert r.status_code == 201, r.text
    return r.json(), best


def _actual_from_plan(plan: dict, *, scale=1.0, skip: tuple = ()) -> list[dict]:
    out = []
    for a in plan["actions"]:
        out.append({
            "plane": a["plane"], "kind": a["kind"],
            "hole_angle": a["hole_angle"],
            "mass": a["mass"] * scale,
            "executed": (a["plane"], a["kind"], a["hole_angle"]) not in skip,
        })
    return out


def test_verification_itemized_match(client):
    bid = balanced(client)
    plan, best = _confirm_first_candidate(client, bid, mixed_planes())

    # 按方案实际执行后的真实振动
    w = np.zeros(2, dtype=complex)
    idx = {"P1": 0, "P2": 1}
    for a in plan["actions"]:
        sign = 1.0 if a["kind"] == "add" else -1.0
        w[idx[a["plane"]]] += sign * a["mass"] * np.exp(1j * np.deg2rad(a["hole_angle"]))
    v_meas = ALPHA_TRUE @ (UNBALANCE + w)

    r = client.post(f"/api/mixed-corrections/{plan['id']}/verifications", json={
        "speed": REF_SPEED,
        "measurements": meas(v_meas),
        "actual_actions": _actual_from_plan(plan),
    })
    assert r.status_code == 201, r.text
    rec = r.json()["reconciliation"]
    assert rec["action_check_verdict"] == "match"
    assert rec["missing_actions"] == []
    assert rec["unmatched_actual_actions"] == []
    for aa in r.json()["actual_actions"]:
        assert aa["status"] == "ok"
        assert abs(aa["mass_deviation"]) < 1e-9
    for s in SENSORS:
        e = rec["sensors"][s]
        assert e["relative_deviation"] < 0.05
        assert e["reduction_ratio"] > 0.8


def test_verification_reports_missing_and_deviation(client):
    bid = balanced(client)
    plan, _ = _confirm_first_candidate(client, bid, mixed_planes())
    actions = plan["actions"]
    assert len(actions) >= 2

    # 第一个动作根本未录入 -> missing；另一个动作质量加倍 -> 偏差
    actual = _actual_from_plan(plan)[1:]
    actual[0]["mass"] = actual[0]["mass"] * 2 + 0.001

    # 重建部分执行后的真实振动
    w = np.zeros(2, dtype=complex)
    idx = {"P1": 0, "P2": 1}
    for a in actual:
        if not a["executed"]:
            continue
        sign = 1.0 if a["kind"] == "add" else -1.0
        w[idx[a["plane"]]] += sign * a["mass"] * np.exp(1j * np.deg2rad(a["hole_angle"]))
    v_meas = ALPHA_TRUE @ (UNBALANCE + w)

    r = client.post(f"/api/mixed-corrections/{plan['id']}/verifications", json={
        "speed": REF_SPEED,
        "measurements": meas(v_meas),
        "actual_actions": actual,
    })
    assert r.status_code == 201, r.text
    rec = r.json()["reconciliation"]
    assert rec["action_check_verdict"] == "mismatch"
    assert len(rec["missing_actions"]) == 1
    assert rec["missing_actions"][0]["hole_angle"] == actions[0]["hole_angle"]
    # 单面合成偏差矢量
    assert any(
        rec["plane_vector_deviation"][p]["magnitude"] > 0 for p in ("P1", "P2"))


def test_verification_flags_not_executed_as_deviation(client):
    bid = balanced(client)
    plan, _ = _confirm_first_candidate(client, bid, mixed_planes())
    actions = plan["actions"]
    assert actions

    # 逐项录入但第一项标记未执行：不是缺失，而是偏差
    skip = (actions[0]["plane"], actions[0]["kind"], actions[0]["hole_angle"])
    actual = _actual_from_plan(plan, skip=(skip,))

    w = np.zeros(2, dtype=complex)
    idx = {"P1": 0, "P2": 1}
    for a in actual:
        if not a["executed"]:
            continue
        sign = 1.0 if a["kind"] == "add" else -1.0
        w[idx[a["plane"]]] += sign * a["mass"] * np.exp(1j * np.deg2rad(a["hole_angle"]))
    v_meas = ALPHA_TRUE @ (UNBALANCE + w)

    r = client.post(f"/api/mixed-corrections/{plan['id']}/verifications", json={
        "speed": REF_SPEED,
        "measurements": meas(v_meas),
        "actual_actions": actual,
    })
    assert r.status_code == 201, r.text
    rec = r.json()["reconciliation"]
    assert rec["missing_actions"] == []
    assert rec["action_check_verdict"] == "deviation"
    first = next(
        a for a in r.json()["actual_actions"]
        if (a["plane"], a["kind"], a["hole_angle"]) == skip)
    assert first["status"] == "not_executed"
    assert abs(first["mass_deviation"] + first["planned_mass"]) < 1e-9


def test_verification_unmatched_actual_action(client):
    bid = balanced(client)
    plan, _ = _confirm_first_candidate(client, bid, mixed_planes())
    w = np.zeros(2, dtype=complex)
    v_meas = V0
    actual = _actual_from_plan(plan) + [{
        "plane": "P1", "kind": "add", "hole_angle": 120.0, "mass": 5,
        "executed": True,
    }]
    r = client.post(f"/api/mixed-corrections/{plan['id']}/verifications", json={
        "speed": REF_SPEED, "measurements": meas(v_meas),
        "actual_actions": actual,
    })
    assert r.status_code == 201, r.text
    rec = r.json()["reconciliation"]
    assert rec["action_check_verdict"] == "mismatch"
    assert rec["unmatched_actual_actions"][0]["hole_angle"] == 120.0


def test_verification_phase_reference_conflict(client):
    bid = balanced(client)
    plan, _ = _confirm_first_candidate(client, bid, mixed_planes())
    r = client.post(f"/api/mixed-corrections/{plan['id']}/verifications", json={
        "speed": REF_SPEED, "phase_reference": "lead",
        "measurements": meas(V0),
        "actual_actions": _actual_from_plan(plan),
    })
    assert r.status_code == 422
    assert r.json()["error"] == "phase_reference_conflict"


# ------------------------------------------------------------ 轴跳档案沿用


def test_mixed_uses_selected_runout_profile(client):
    bid = make_batch(client)
    runout = np.array([3.0 * np.exp(1j * np.deg2rad(100)),
                       2.0 * np.exp(1j * np.deg2rad(20))])
    v0 = ALPHA_TRUE @ UNBALANCE
    client.post(f"/api/batches/{bid}/runs", json={
        "kind": "baseline", "speed": REF_SPEED, "measurements": meas(runout + v0)})
    client.post(f"/api/batches/{bid}/runs", json={
        "kind": "trial", "speed": REF_SPEED,
        "measurements": meas(runout + v0 + ALPHA_TRUE @ T1),
        "trial_weights": [{"plane": "P1", "mass": 20, "angle": 0}]})
    client.post(f"/api/batches/{bid}/runs", json={
        "kind": "trial", "speed": REF_SPEED,
        "measurements": meas(runout + v0 + ALPHA_TRUE @ T2),
        "trial_weights": [{"plane": "P2", "mass": 20, "angle": 90}]})
    prof = client.post(f"/api/batches/{bid}/runout-profiles", json={
        "name": "慢转-A", "slow_roll_speed_limit": 600.0,
        "dispersion_limit": 0.5,
        "records": [
            {"speed": 200.0, "measurements": meas(runout)} for _ in range(2)
        ],
    }).json()
    cal = client.post(f"/api/batches/{bid}/calibrate",
                      json={"runout_profile_id": prof["id"]})
    assert cal.status_code == 200, cal.text

    # 不带档案搜索：不能错误复用带档案标定
    r = search(client, bid, mixed_planes())
    assert r.status_code == 422
    assert r.json()["error"] == "calibration_missing"

    r = search(client, bid, mixed_planes(), runout_profile_id=prof["id"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["runout_profile_id"] == prof["id"]
    best = body["candidates"][0]
    assert best["predicted_metric"] < 0.2 * float(np.max(np.abs(v0)))

    # 方案快照内嵌基线轴跳扣除
    plan, _ = _confirm_first_candidate(
        client, bid, mixed_planes(), runout_profile_id=prof["id"])
    assert plan["runout_compensation"]["runout_profile_id"] == prof["id"]

    # 复测沿用同一档案
    w = np.zeros(2, dtype=complex)
    idx = {"P1": 0, "P2": 1}
    for a in plan["actions"]:
        sign = 1.0 if a["kind"] == "add" else -1.0
        w[idx[a["plane"]]] += sign * a["mass"] * np.exp(1j * np.deg2rad(a["hole_angle"]))
    v_meas = runout + ALPHA_TRUE @ (UNBALANCE + w)
    ver = client.post(f"/api/mixed-corrections/{plan['id']}/verifications", json={
        "speed": REF_SPEED, "measurements": meas(v_meas),
        "actual_actions": _actual_from_plan(plan),
    })
    assert ver.status_code == 201, ver.text
    rec = ver.json()["reconciliation"]
    assert rec["runout_profile"]["runout_profile_id"] == prof["id"]
    for s in SENSORS:
        assert rec["sensors"][s]["relative_deviation"] < 0.05


def test_mixed_phase_reference_conflict_with_runout(client):
    bid = make_batch(client)
    runout = np.array([3.0 + 0j, 2.0 + 0j])
    v0 = ALPHA_TRUE @ UNBALANCE
    client.post(f"/api/batches/{bid}/runs", json={
        "kind": "baseline", "speed": REF_SPEED, "phase_reference": "lag",
        "measurements": meas(runout + v0)})
    client.post(f"/api/batches/{bid}/runs", json={
        "kind": "trial", "speed": REF_SPEED, "phase_reference": "lag",
        "measurements": meas(runout + v0 + ALPHA_TRUE @ T1),
        "trial_weights": [{"plane": "P1", "mass": 20, "angle": 0}]})
    client.post(f"/api/batches/{bid}/runs", json={
        "kind": "trial", "speed": REF_SPEED, "phase_reference": "lag",
        "measurements": meas(runout + v0 + ALPHA_TRUE @ T2),
        "trial_weights": [{"plane": "P2", "mass": 20, "angle": 90}]})
    prof = client.post(f"/api/batches/{bid}/runout-profiles", json={
        "name": "lead档案", "slow_roll_speed_limit": 600.0,
        "dispersion_limit": 0.5,
        "records": [{"speed": 200.0, "phase_reference": "lead",
                     "measurements": meas(runout)}],
    }).json()
    # lead 档案不能标定 lag 运行
    r = client.post(f"/api/batches/{bid}/calibrate",
                    json={"runout_profile_id": prof["id"]})
    assert r.status_code == 422
    assert r.json()["error"] == "runout_phase_reference_conflict"


# ------------------------------------------------------------ 隔离与导出


def test_mixed_flow_does_not_modify_existing_data(client):
    bid = balanced(client)
    sols_before = client.post(f"/api/batches/{bid}/solutions",
                              json={"top": 3}).json()
    cal_before = client.get(f"/api/batches/{bid}/export").json()["calibrations"]

    search(client, bid, mixed_planes())
    plan, _ = _confirm_first_candidate(client, bid, mixed_planes())
    client.post(f"/api/mixed-corrections/{plan['id']}/verifications", json={
        "speed": REF_SPEED, "measurements": meas(V0),
        "actual_actions": _actual_from_plan(plan),
    })

    rec = client.get(f"/api/batches/{bid}/export").json()
    # 标定未被新增/改写（混合流程不自动标定）
    assert len(rec["calibrations"]) == len(cal_before)
    # 既有运行与加重方案原样保留
    assert len(rec["runs"]) == 3
    existing_ids = {s["id"] for s in sols_before["discrete"]} | \
        {sols_before["continuous"]["id"]}
    assert {s["id"] for s in rec["solutions"]} == existing_ids
    # 混合方案进入导出
    assert len(rec["mixed_plans"]) == 1
    mp = rec["mixed_plans"][0]
    assert mp["id"] == plan["id"]
    assert mp["geometry_snapshot"]["planes"]
    assert mp["verifications"][0]["reconciliation"]["sensors"]
    # 轴跳使用位置包含混合方案键
    assert "mixed_plan_ids" in rec["runout_usage"][0] if rec["runout_usage"] else True


def test_snapshot_independent_of_later_geometry(client):
    bid = balanced(client)
    sr = search(client, bid, mixed_planes())
    body = sr.json()
    best = body["candidates"][0]
    r = client.post(f"/api/batches/{bid}/mixed-corrections", json={
        "planes": _planes_from_search(body),
        "removal_step": body["removal_step"],
        "actions": [
            {"plane": a["plane"], "kind": a["kind"],
             "hole_angle": a["hole_angle"], "mass": a["mass"]}
            for a in best["actions"]
        ],
    })
    plan = r.json()
    # 快照厚度固定为录入值
    g0 = plan["geometry_snapshot"]["planes"][0]
    assert g0["drill_holes"][0]["current_thickness"] == 10.0
    assert g0["drill_holes"][0]["min_remaining_thickness"] == 2.0
    # 重新用更严的厚度/上限搜索不影响已存方案
    tighter = mixed_planes(P1={
        "drill_holes": [
            {"hole_angle": a, "current_thickness": 6.0, "removal_limit": 20.0,
             "min_remaining_thickness": 5.0}
            for a in HOLES
        ]
    })
    r2 = search(client, bid, tighter)
    assert r2.status_code == 200
    again = client.get(f"/api/mixed-corrections/{plan['id']}").json()
    assert again["geometry_snapshot"]["planes"][0]["drill_holes"][0][
        "current_thickness"] == 10.0
