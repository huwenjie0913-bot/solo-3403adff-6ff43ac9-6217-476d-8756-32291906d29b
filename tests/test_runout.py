"""慢转轴跳补偿：建档/统计、扣除补偿、拒绝分支、不可判定、切换隔离与导出。"""

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
RUNOUT = np.array([
    3.0 * np.exp(1j * np.deg2rad(100)),
    2.0 * np.exp(1j * np.deg2rad(20)),
])
SENSORS = ["DE", "NDE"]
REF_SPEED = 3000.0
T1 = np.array([20 * np.exp(1j * np.deg2rad(0)), 0])
T2 = np.array([0, 20 * np.exp(1j * np.deg2rad(90))])


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
        "name": "汽轮机-慢转轴跳",
        "reference_speed": REF_SPEED,
        "speed_tolerance": 30.0,
        "planes": [
            {"name": "P1", "correction_radius": 250.0,
             "hole_angles": [i * 30 for i in range(12)], "mass_limit": 200.0},
            {"name": "P2", "correction_radius": 250.0,
             "hole_angles": [i * 30 for i in range(12)], "mass_limit": 200.0},
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


def add_run(client, bid, kind, v, trial_weights=None, speed=REF_SPEED, phase_ref="lag"):
    return client.post(f"/api/batches/{bid}/runs", json={
        "kind": kind, "speed": speed, "phase_reference": phase_ref,
        "measurements": meas(v), "trial_weights": trial_weights or [],
    })


def balanced_with_runout(client, **batch_overrides) -> tuple[int, np.ndarray]:
    """基线与试重测量都含轴跳 RUNOUT。返回 (batch_id, 真实净基线)。"""
    bid = make_batch(client, **batch_overrides)
    v0 = ALPHA_TRUE @ UNBALANCE
    add_run(client, bid, "baseline", RUNOUT + v0)
    add_run(client, bid, "trial", RUNOUT + v0 + ALPHA_TRUE @ T1,
            [{"plane": "P1", "mass": 20, "angle": 0}])
    add_run(client, bid, "trial", RUNOUT + v0 + ALPHA_TRUE @ T2,
            [{"plane": "P2", "mass": 20, "angle": 90}])
    return bid, v0


def make_profile(client, bid, vectors=None, *, speed=200.0, limit=600.0,
                 dispersion_limit=0.5, phase_ref="lag", name="慢转档案-A"):
    vectors = vectors if vectors is not None else [RUNOUT] * 3
    payload = {
        "name": name,
        "slow_roll_speed_limit": limit,
        "dispersion_limit": dispersion_limit,
        "records": [
            {"speed": speed, "phase_reference": phase_ref, "measurements": meas(v)}
            for v in vectors
        ],
    }
    r = client.post(f"/api/batches/{bid}/runout-profiles", json=payload)
    assert r.status_code == 201, r.text
    return r.json()


# ------------------------------------------------------------ 档案与统计


def test_runout_profile_aggregates_complex_vectors(client):
    bid = make_batch(client)
    r0 = RUNOUT
    records_vec = [
        r0,
        r0 * (1.0 * np.exp(1j * np.deg2rad(1.0))),
        np.array([r0[0] * 0.98, r0[1] * 1.02]),
    ]
    prof = make_profile(client, bid, records_vec, dispersion_limit=1.0)

    assert prof["usable"] is True
    assert prof["issues"] == []
    for i, s in enumerate(SENSORS):
        st = prof["summary"][s]
        mean = complex(st["real"], st["imag"])
        expected = np.mean([v[i] for v in records_vec])
        assert abs(mean - expected) < 1e-9
        assert st["dispersion"] == pytest.approx(
            np.sqrt(np.mean(np.abs(np.array([v[i] for v in records_vec]) - expected) ** 2))
        )
        assert st["record_count"] == 3
        assert st["mean"]["amplitude"] == pytest.approx(abs(expected))

    # 批次列表带出档案数量
    listing = client.get("/api/batches").json()
    entry = next(b for b in listing if b["id"] == bid)
    assert entry["runout_profile_count"] == 1


def test_append_record_recomputes_summary(client):
    bid = make_batch(client)
    prof = make_profile(client, bid, [RUNOUT])
    pid = prof["id"]
    assert prof["summary"]["DE"]["record_count"] == 1

    r = client.post(
        f"/api/batches/{bid}/runout-profiles/{pid}/records",
        json={"speed": 210.0, "measurements": meas(RUNOUT)},
    )
    assert r.status_code == 201, r.text
    assert r.json()["summary"]["DE"]["record_count"] == 2
    assert r.json()["usable"] is True


def test_profile_overspeed_and_missing_sensor_marked_and_rejected(client):
    bid, _ = balanced_with_runout(client)

    # 一条超速 + 一条缺 NDE 测点
    payload = {
        "name": "坏档案",
        "slow_roll_speed_limit": 300.0,
        "dispersion_limit": 0.5,
        "records": [
            {"speed": 350.0, "measurements": meas(RUNOUT)},
            {"speed": 200.0,
             "measurements": [{"sensor": "DE", "amplitude": abs(RUNOUT[0]),
                               "phase": np.rad2deg(np.angle(RUNOUT[0])) % 360}]},
        ],
    }
    r = client.post(f"/api/batches/{bid}/runout-profiles", json=payload)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["usable"] is False
    codes = {i["code"] for i in body["issues"]}
    assert {"overspeed", "sensor_coverage"} <= codes
    overspeed = next(i for i in body["issues"] if i["code"] == "overspeed")
    assert overspeed["records"][0]["speed"] == 350.0
    coverage = next(i for i in body["issues"] if i["code"] == "sensor_coverage")
    assert coverage["records"][0]["missing_sensors"] == ["NDE"]

    # 拒绝用于标定，指出记录与测点
    r = client.post(f"/api/batches/{bid}/calibrate",
                    json={"runout_profile_id": body["id"]})
    assert r.status_code == 422
    err = r.json()
    assert err["error"] == "runout_profile_invalid"
    assert err["details"]["runout_profile_id"] == body["id"]
    assert {i["code"] for i in err["details"]["issues"]} >= {"overspeed", "sensor_coverage"}


def test_dispersion_exceeded_points_to_sensor(client):
    bid, _ = balanced_with_runout(client)
    bad_vecs = [
        RUNOUT,
        np.array([0.5 * RUNOUT[0], RUNOUT[1] * np.exp(1j * np.deg2rad(40))]),
    ]
    prof = make_profile(client, bid, bad_vecs, dispersion_limit=0.2)
    assert prof["usable"] is False
    issue = next(i for i in prof["issues"] if i["code"] == "dispersion_exceeded")
    assert issue["dispersion_limit"] == 0.2
    sensors_flagged = {s["sensor"] for s in issue["sensors"]}
    assert sensors_flagged == {"DE", "NDE"}

    r = client.post(f"/api/batches/{bid}/solutions",
                    json={"runout_profile_id": prof["id"]})
    assert r.status_code == 422
    assert r.json()["error"] == "runout_profile_invalid"


def test_duplicate_and_unknown_sensor_in_record(client):
    bid = make_batch(client)
    payload = {
        "name": "坏档案2",
        "slow_roll_speed_limit": 600.0,
        "dispersion_limit": 0.5,
        "records": [
            {"speed": 200.0, "measurements": [
                {"sensor": "DE", "amplitude": 1.0, "phase": 0.0},
                {"sensor": "DE", "amplitude": 1.0, "phase": 0.0},
                {"sensor": "XX", "amplitude": 1.0, "phase": 0.0},
                {"sensor": "NDE", "amplitude": 1.0, "phase": 0.0},
            ]},
        ],
    }
    r = client.post(f"/api/batches/{bid}/runout-profiles", json=payload)
    body = r.json()
    issue = next(i for i in body["issues"] if i["code"] == "sensor_coverage")
    rec = issue["records"][0]
    assert rec["duplicate_sensors"] == ["DE"]
    assert rec["unknown_sensors"] == ["XX"]
    assert body["usable"] is False


def test_runout_phase_reference_conflict(client):
    bid, _ = balanced_with_runout(client)
    # 运行均为 lag；档案为 lead
    prof = make_profile(client, bid, [RUNOUT], phase_ref="lead")
    r = client.post(f"/api/batches/{bid}/calibrate",
                    json={"runout_profile_id": prof["id"]})
    assert r.status_code == 422
    err = r.json()
    assert err["error"] == "runout_phase_reference_conflict"
    assert err["details"]["runout_phase_references"] == ["lead"]
    assert err["details"]["run_phase_references"] == ["lag"]


# ------------------------------------------------------------ 扣除后计算


def test_solve_without_profile_does_not_reuse_profile_calibration(client):
    bid, _ = balanced_with_runout(client)
    prof = make_profile(client, bid)

    client.post(f"/api/batches/{bid}/calibrate",
                json={"runout_profile_id": prof["id"]})
    # 不带档案求解：应自动建立无档案标定，而非复用最近的带档案标定
    r = client.post(f"/api/batches/{bid}/solutions", json={"top": 3})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["continuous"]["runout_profile_id"] is None
    assert body["continuous"]["runout_compensation"] is None
    export = client.get(f"/api/batches/{bid}/export").json()
    assert {c["runout_profile_id"] for c in export["calibrations"]} == {None, prof["id"]}


def test_calibrate_and_solve_with_runout(client):
    bid, v0 = balanced_with_runout(client)
    prof = make_profile(client, bid)
    pid = prof["id"]

    r = client.post(f"/api/batches/{bid}/calibrate",
                    json={"runout_profile_id": pid})
    assert r.status_code == 200, r.text
    cal = r.json()
    assert cal["runout_profile_id"] == pid
    # 差分抵消轴跳，影响系数仍可精确还原
    for i, s in enumerate(SENSORS):
        for j, p in enumerate(["P1", "P2"]):
            z = complex(cal["coefficients"][s][p]["real"],
                        cal["coefficients"][s][p]["imag"])
            assert abs(z - ALPHA_TRUE[i, j]) < 1e-9

    block = cal["runout_compensation"]
    assert block["runout_profile_id"] == pid
    assert block["summary"] == prof["summary"]
    base_item = next(it for it in block["runs"] if it["kind"] == "baseline")
    for i, s in enumerate(SENSORS):
        e = base_item["sensors"][s]
        assert e["raw"]["amplitude"] == pytest.approx(abs(RUNOUT[i] + v0[i]))
        assert e["compensation"]["amplitude"] == pytest.approx(abs(RUNOUT[i]))
        assert e["net"]["amplitude"] == pytest.approx(abs(v0[i]), abs=1e-9)

    r = client.post(f"/api/batches/{bid}/solutions",
                    json={"top": 5, "runout_profile_id": pid})
    assert r.status_code == 201, r.text
    body = r.json()
    cont = body["continuous"]
    assert cont["runout_profile_id"] == pid
    for w, u in zip(cont["weights"], UNBALANCE):
        assert w["mass"] == pytest.approx(abs(u), abs=1e-6)
        assert w["angle"] == pytest.approx(
            (np.rad2deg(np.angle(-u))) % 360, abs=1e-6)
    assert cont["predicted_metric"] < 1e-6
    assert cont["runout_compensation"]["runout_profile_id"] == pid
    # 连续解把净残振降到误差界以内：标为不可判定（不得当作“平衡到零”）
    assert set(cont["resolution"]["undecidable_sensors"]) == {"DE", "NDE"}
    for s in SENSORS:
        assert cont["resolution"]["sensors"][s]["undecidable"] is True
        assert cont["resolution"]["sensors"][s]["predicted_amplitude"] <= \
            cont["resolution"]["sensors"][s]["resolution"]


def test_switching_profile_keeps_history(client):
    bid, v0 = balanced_with_runout(client)
    prof = make_profile(client, bid, name="档案-A")

    # 先用档案求解并复测（锁定为历史）
    sol_a = client.post(f"/api/batches/{bid}/solutions",
                        json={"runout_profile_id": prof["id"], "top": 3}).json()
    best = sol_a["discrete"][0]
    w = np.array([
        best["weights"][0]["mass"] * np.exp(1j * np.deg2rad(best["weights"][0]["angle"])),
        best["weights"][1]["mass"] * np.exp(1j * np.deg2rad(best["weights"][1]["angle"])),
    ])
    v_meas = RUNOUT + ALPHA_TRUE @ (UNBALANCE + w)
    r = client.post(f"/api/solutions/{best['id']}/verifications",
                    json={"speed": REF_SPEED, "measurements": meas(v_meas)})
    assert r.status_code == 201, r.text

    # 无档案重新求解：预测针对含轴跳的原始振动；不删除档案方案
    r = client.post(f"/api/batches/{bid}/solutions", json={"top": 3})
    assert r.status_code == 201, r.text
    sols = client.get(f"/api/batches/{bid}/solutions").json()
    old_ids = {sol_a["continuous"]["id"]} | {s["id"] for s in sol_a["discrete"]}
    assert best["id"] in {s["id"] for s in sols}
    # 本轮无档案求解产生的新方案不携带轴跳档案
    assert all(s["runout_profile_id"] is None for s in sols
               if s["id"] not in old_ids)

    # 无档案连续解把轴跳误当不平衡：所需配重明显更大
    plain_cont = next(s for s in sols if s["kind"] == "continuous"
                      and s["runout_profile_id"] is None)
    comp_cont = sol_a["continuous"]
    assert plain_cont["total_mass"] > comp_cont["total_mass"] + 1.0

    # 标定也保留两份，且不改动原始运行
    export = client.get(f"/api/batches/{bid}/export").json()
    assert len(export["calibrations"]) == 2
    cal_pids = {c["runout_profile_id"] for c in export["calibrations"]}
    assert cal_pids == {None, prof["id"]}
    assert all(run["measurements"] for run in export["runs"])


# ------------------------------------------------------------ 复测与不可判定


def test_verification_subtracts_runout(client):
    bid, v0 = balanced_with_runout(client, amp_error=0.001, phase_error=0.1)
    prof = make_profile(client, bid)
    sol = client.post(f"/api/batches/{bid}/solutions",
                      json={"runout_profile_id": prof["id"], "top": 3}).json()
    best = sol["discrete"][0]
    w = np.array([
        best["weights"][0]["mass"] * np.exp(1j * np.deg2rad(best["weights"][0]["angle"])),
        best["weights"][1]["mass"] * np.exp(1j * np.deg2rad(best["weights"][1]["angle"])),
    ])
    v_meas = RUNOUT + ALPHA_TRUE @ (UNBALANCE + w)
    r = client.post(f"/api/solutions/{best['id']}/verifications",
                    json={"speed": REF_SPEED, "measurements": meas(v_meas)})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["runout_profile_id"] == prof["id"]
    comp = body["comparison"]
    assert comp["balance_verdict"] == "decidable"
    for i, s in enumerate(SENSORS):
        e = comp["sensors"][s]
        assert e["raw_measured"]["amplitude"] == pytest.approx(abs(v_meas[i]))
        assert e["runout_compensation"]["amplitude"] == pytest.approx(abs(RUNOUT[i]))
        assert e["measured"]["amplitude"] == pytest.approx(
            abs(ALPHA_TRUE @ (UNBALANCE + w))[i], abs=1e-6)
        assert e["relative_deviation"] < 0.05
        assert e["reduction_ratio"] > 0.8
        assert e["verdict"] == "decidable"


def test_verification_undecidable_when_net_within_resolution(client):
    # 残余不平衡极小、净振动落入误差界；轴跳很大 -> 原始幅值看似“超标”，
    # 扣除后必须标记不可判定，不得当作达标
    bid = make_batch(client)
    v0 = np.array([0.02 + 0j, 0.01 + 0j])  # 净基线几乎为 0
    add_run(client, bid, "baseline", RUNOUT + v0)
    add_run(client, bid, "trial", RUNOUT + v0 + ALPHA_TRUE @ T1,
            [{"plane": "P1", "mass": 20, "angle": 0}])
    add_run(client, bid, "trial", RUNOUT + v0 + ALPHA_TRUE @ T2,
            [{"plane": "P2", "mass": 20, "angle": 90}])
    prof = make_profile(client, bid)
    sol = client.post(f"/api/batches/{bid}/solutions",
                      json={"runout_profile_id": prof["id"], "top": 3}).json()
    cont = sol["continuous"]
    assert set(cont["resolution"]["undecidable_sensors"]) == {"DE", "NDE"}

    best = sol["discrete"][0]
    # 复测：装了方案后真实残余依旧极小，但原始测量几乎全是轴跳
    v_meas = RUNOUT + v0 * 0.5
    r = client.post(f"/api/solutions/{best['id']}/verifications",
                    json={"speed": REF_SPEED, "measurements": meas(v_meas)})
    assert r.status_code == 201, r.text
    comp = r.json()["comparison"]
    assert comp["balance_verdict"] == "undecidable"
    assert set(comp["undecidable_sensors"]) == {"DE", "NDE"}
    for s in SENSORS:
        e = comp["sensors"][s]
        assert e["undecidable"] is True
        assert e["verdict"] == "undecidable"
        assert e["raw_measured"]["amplitude"] > 1.0
        assert e["measured"]["amplitude"] < e["resolution"]
        # 不可判定时降幅必须置空，不能报“达标降幅”
        assert e.get("reduction_ratio") is None


def test_verification_phase_reference_conflict(client):
    bid, _ = balanced_with_runout(client)
    prof = make_profile(client, bid)
    sol = client.post(f"/api/batches/{bid}/solutions",
                      json={"runout_profile_id": prof["id"], "top": 3}).json()
    best = sol["discrete"][0]
    r = client.post(f"/api/solutions/{best['id']}/verifications", json={
        "speed": REF_SPEED, "phase_reference": "lead",
        "measurements": meas(RUNOUT),
    })
    assert r.status_code == 422
    assert r.json()["error"] == "phase_reference_conflict"

    # 批次基准 lag；显式指定 lead 档案用于复测（该方案未用档案）
    lead_prof = make_profile(client, bid, [RUNOUT], phase_ref="lead", name="档案-lead")
    sol_plain = client.post(f"/api/batches/{bid}/solutions", json={"top": 3}).json()
    r = client.post(f"/api/solutions/{sol_plain['continuous']['id']}/verifications", json={
        "speed": REF_SPEED, "runout_profile_id": lead_prof["id"],
        "measurements": meas(RUNOUT),
    })
    assert r.status_code == 422
    assert r.json()["error"] == "runout_phase_reference_conflict"


# ------------------------------------------------------------ 导出快照


def test_export_keeps_profile_snapshot_and_usage(client):
    bid, _ = balanced_with_runout(client)
    prof = make_profile(client, bid)
    pid = prof["id"]
    client.post(f"/api/batches/{bid}/calibrate",
                json={"runout_profile_id": pid})
    sol = client.post(f"/api/batches/{bid}/solutions",
                      json={"runout_profile_id": pid, "top": 3}).json()
    best = sol["discrete"][0]
    client.post(f"/api/solutions/{best['id']}/verifications",
                json={"speed": REF_SPEED, "measurements": meas(RUNOUT)})

    rec = client.get(f"/api/batches/{bid}/export").json()
    assert len(rec["runout_profiles"]) == 1
    snap = rec["runout_profiles"][0]
    assert snap["id"] == pid
    assert snap["slow_roll_speed_limit"] == 600.0
    assert snap["dispersion_limit"] == 0.5
    assert len(snap["records"]) == 3
    assert snap["summary"]["DE"]["record_count"] == 3

    usage = next(u for u in rec["runout_usage"] if u["runout_profile_id"] == pid)
    assert usage["calibration_ids"]
    assert best["id"] in usage["solution_ids"]
    assert usage["verification_ids"]

    # 各使用点内嵌实际使用快照
    used_cal = next(c for c in rec["calibrations"] if c["runout_profile_id"] == pid)
    assert used_cal["runout_compensation"]["snapshotted_at"]
    used_sol = next(s for s in rec["solutions"] if s["id"] == best["id"])
    assert used_sol["runout_compensation"]["runout_profile_id"] == pid
    ver = used_sol["verifications"][0]
    assert ver["runout_profile_id"] == pid
    assert ver["comparison"]["runout_profile"]["summary"]
