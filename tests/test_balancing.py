"""端到端测试：合成转子 -> 标定 -> 求解 -> 复测 -> 导出，以及各错误分支。"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.vibration import complex_to_amp_phase
from app.main import app

# ------------------------------------------------------------ 合成机器
# 真实影响系数 (2 测点 x 2 面)，单位 mm/s 每克
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

V0 = ALPHA_TRUE @ UNBALANCE
T1 = np.array([20 * np.exp(1j * np.deg2rad(0)), 0])    # P1 试重 20g∠0°
T2 = np.array([0, 20 * np.exp(1j * np.deg2rad(90))])   # P2 试重 20g∠90°


def meas(v: np.ndarray) -> list[dict]:
    out = []
    for name, z in zip(["DE", "NDE"], v):
        amp, phase = complex_to_amp_phase(z)
        out.append({"sensor": name, "amplitude": amp, "phase": phase})
    return out


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def make_batch(client: TestClient, **overrides) -> int:
    payload = {
        "name": "风机-01",
        "description": "送风机双面现场动平衡",
        "reference_speed": REF_SPEED,
        "speed_tolerance": 30.0,
        "planes": [
            {"name": "P1", "correction_radius": 250.0,
             "hole_angles": [i * 30 for i in range(12)], "mass_limit": 100.0},
            {"name": "P2", "correction_radius": 250.0,
             "hole_angles": [i * 30 for i in range(12)], "mass_limit": 100.0},
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


def add_run(client, batch_id, kind, v, trial_weights=None, speed=REF_SPEED, phase_ref="lag"):
    payload = {
        "kind": kind,
        "speed": speed,
        "phase_reference": phase_ref,
        "measurements": meas(v),
        "trial_weights": trial_weights or [],
    }
    return client.post(f"/api/batches/{batch_id}/runs", json=payload)


def balanced_batch(client) -> int:
    """建立含基线 + 两次试重的批次。"""
    bid = make_batch(client)
    assert add_run(client, bid, "baseline", V0).status_code == 201
    assert add_run(client, bid, "trial", V0 + ALPHA_TRUE @ T1,
                   [{"plane": "P1", "mass": 20, "angle": 0}]).status_code == 201
    assert add_run(client, bid, "trial", V0 + ALPHA_TRUE @ T2,
                   [{"plane": "P2", "mass": 20, "angle": 90}]).status_code == 201
    return bid


# ------------------------------------------------------------ 正常流程


def test_calibrate_recovers_coefficients(client):
    bid = balanced_batch(client)
    r = client.post(f"/api/batches/{bid}/calibrate")
    assert r.status_code == 200, r.text
    body = r.json()

    for i, s in enumerate(["DE", "NDE"]):
        for j, p in enumerate(["P1", "P2"]):
            got = body["coefficients"][s][p]
            z = complex(got["real"], got["imag"])
            assert abs(z - ALPHA_TRUE[i, j]) < 1e-9
            assert got["magnitude"] == pytest.approx(abs(ALPHA_TRUE[i, j]))

    # 无噪声数据拟合残差应接近 0，且带来源信息
    assert all(res["amplitude"] < 1e-9 for res in body["residuals"])
    prov = body["provenance"]
    assert prov["sensors"] == ["DE", "NDE"]
    assert prov["planes"] == ["P1", "P2"]
    assert prov["baseline_run_id"] and len(prov["trial_run_ids"]) == 2
    assert body["condition"] < 10


def test_solve_continuous_and_discrete(client):
    bid = balanced_batch(client)
    r = client.post(f"/api/batches/{bid}/solutions", json={"top": 5})
    assert r.status_code == 201, r.text
    body = r.json()

    cont = body["continuous"]
    assert cont["kind"] == "continuous"
    # 连续解应还原 -不平衡量
    for w, u in zip(cont["weights"], UNBALANCE):
        assert w["mass"] == pytest.approx(abs(u), abs=1e-6)
        assert w["angle"] == pytest.approx((np.rad2deg(np.angle(-u))) % 360, abs=1e-6)
    assert cont["predicted_metric"] < 1e-6
    assert cont["worst_case"] >= cont["predicted_metric"]

    discrete = body["discrete"]
    assert len(discrete) == 5
    # 排序：预测残振非降
    metrics = [s["predicted_metric"] for s in discrete]
    assert metrics == sorted(metrics)
    # 最优离散组合应显著降低振动
    baseline_metric = float(np.max(np.abs(V0)))
    assert metrics[0] < 0.15 * baseline_metric
    # 离散解给出孔位安装明细，且每面总质量不超上限
    best = discrete[0]
    for w in best["weights"]:
        assert w["assignments"], "离散解应包含安装明细"
        assert sum(a["mass"] for a in w["assignments"]) <= 100.0 + 1e-9
        for a in w["assignments"]:
            assert a["hole_angle"] % 30 == pytest.approx(0)
    assert all(s["worst_case"] >= s["predicted_metric"] for s in discrete)


def test_verification_and_export(client):
    bid = balanced_batch(client)
    sol = client.post(f"/api/batches/{bid}/solutions", json={"top": 3}).json()
    best = sol["discrete"][0]

    # 用合成机器仿真安装最优离散方案后的实测振动
    w = np.array([
        best["weights"][0]["mass"] * np.exp(1j * np.deg2rad(best["weights"][0]["angle"])),
        best["weights"][1]["mass"] * np.exp(1j * np.deg2rad(best["weights"][1]["angle"])),
    ])
    v_meas = ALPHA_TRUE @ (UNBALANCE + w)

    r = client.post(f"/api/solutions/{best['id']}/verifications",
                    json={"speed": REF_SPEED, "measurements": meas(v_meas)})
    assert r.status_code == 201, r.text
    comp = r.json()["comparison"]
    for s in ["DE", "NDE"]:
        entry = comp["sensors"][s]
        assert entry["relative_deviation"] < 0.05
        assert entry["reduction_ratio"] > 0.8

    # 已复测方案在重新求解后应保留
    client.post(f"/api/batches/{bid}/solutions", json={"top": 3})
    sols = client.get(f"/api/batches/{bid}/solutions").json()
    assert any(s["id"] == best["id"] for s in sols)

    r = client.get(f"/api/batches/{bid}/export")
    assert r.status_code == 200
    rec = r.json()
    assert rec["record_type"] == "two_plane_field_balancing"
    assert rec["batch"]["id"] == bid
    assert len(rec["runs"]) == 3
    assert rec["calibration"]["provenance"]["trial_run_ids"]
    verified = [s for s in rec["solutions"] if s["verifications"]]
    assert verified and verified[0]["verifications"][0]["comparison"]["sensors"]


# ------------------------------------------------------------ 错误分支


def test_speed_deviation_points_to_run(client):
    bid = make_batch(client)
    add_run(client, bid, "baseline", V0)
    r = add_run(client, bid, "trial", V0 + ALPHA_TRUE @ T1,
                [{"plane": "P1", "mass": 20, "angle": 0}], speed=REF_SPEED + 100)
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "speed_deviation"
    assert body["details"]["reference_speed"] == REF_SPEED
    assert body["details"]["runs"][0]["deviation"] == pytest.approx(100.0)


def test_phase_reference_conflict(client):
    bid = make_batch(client)
    add_run(client, bid, "baseline", V0, phase_ref="lag")
    r = add_run(client, bid, "trial", V0 + ALPHA_TRUE @ T1,
                [{"plane": "P1", "mass": 20, "angle": 0}], phase_ref="lead")
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "phase_reference_conflict"
    assert set(body["details"]["references"]) == {"lag", "lead"}


def test_insufficient_trials_identifies_plane(client):
    bid = make_batch(client)
    add_run(client, bid, "baseline", V0)
    add_run(client, bid, "trial", V0 + ALPHA_TRUE @ T1,
            [{"plane": "P1", "mass": 20, "angle": 0}])
    r = client.post(f"/api/batches/{bid}/calibrate")
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "insufficient_trials"
    assert "P2" in body["details"]["planes_without_trials"]


def test_ill_conditioned_matrix(client):
    bid = make_batch(client)
    add_run(client, bid, "baseline", V0)
    t1 = np.array([10.0 + 0j, 10.0 + 0j])
    t2 = np.array([10.0 + 0j, 10 * np.exp(1j * np.deg2rad(0.0001))])
    add_run(client, bid, "trial", V0 + ALPHA_TRUE @ t1,
            [{"plane": "P1", "mass": 10, "angle": 0},
             {"plane": "P2", "mass": 10, "angle": 0}])
    add_run(client, bid, "trial", V0 + ALPHA_TRUE @ t2,
            [{"plane": "P1", "mass": 10, "angle": 0},
             {"plane": "P2", "mass": 10, "angle": 0.0001}])
    r = client.post(f"/api/batches/{bid}/calibrate")
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "ill_conditioned_matrix"
    assert body["details"]["condition"] > body["details"]["threshold"]


def test_no_feasible_combination_identifies_constraint(client):
    # 质量上限 1g 低于最小配重规格 2g -> 无可安装组合
    bid = make_batch(client, planes=[
        {"name": "P1", "correction_radius": 250.0,
         "hole_angles": [i * 30 for i in range(12)], "mass_limit": 1.0},
        {"name": "P2", "correction_radius": 250.0,
         "hole_angles": [i * 30 for i in range(12)], "mass_limit": 1.0},
    ])
    add_run(client, bid, "baseline", V0)
    add_run(client, bid, "trial", V0 + ALPHA_TRUE @ T1,
            [{"plane": "P1", "mass": 20, "angle": 0}])
    add_run(client, bid, "trial", V0 + ALPHA_TRUE @ T2,
            [{"plane": "P2", "mass": 20, "angle": 90}])
    r = client.post(f"/api/batches/{bid}/solutions", json={})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "no_feasible_combination"
    assert body["details"]["mass_limit"] == 1.0
    assert body["details"]["min_weight_spec"] == 2.0


def test_run_validation_errors(client):
    bid = make_batch(client)
    # 未知测点
    r = client.post(f"/api/batches/{bid}/runs", json={
        "kind": "baseline", "speed": REF_SPEED,
        "measurements": [{"sensor": "XX", "amplitude": 1, "phase": 0}],
    })
    assert r.status_code == 422
    assert r.json()["details"]["unknown_sensors"] == ["XX"]
    # 未知校正面
    r = add_run(client, bid, "trial", V0, [{"plane": "P9", "mass": 5, "angle": 0}])
    assert r.status_code == 422
    assert "P9" in r.json()["details"]["unknown_planes"]
    # 不存在的批次
    assert client.get("/api/batches/99999").status_code == 404
