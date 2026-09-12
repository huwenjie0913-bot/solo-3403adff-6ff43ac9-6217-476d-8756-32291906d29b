"""慢转轴跳（slow-roll runout）档案建立与矢量补偿。

轴跳是慢转（低转速）下由机械/电测量原因产生的 1X 假振动矢量。流程：

1. 同一测点的多条慢转记录 z_k = A_k·exp(iφ_k) 换算为复矢量；
2. 取复均值作为该测点的轴跳估计
       mean_s   = Σ_k z_{s,k} / n
   离散度取各次记录对均值的 RMS 偏差
       disp_s   = sqrt( Σ_k |z_{s,k} - mean_s|² / n )
   并相对均值幅值给出重复度比值；
3. 档案须通过：测点齐全且无重复、全部记录不超速、相位基准一致、
   逐测点离散度不超过阈值，方可用于补偿；
4. 补偿即逐测点净振动 z_net = z_raw - runout_s。

可分辨范围沿用批次幅相误差模型（保守一阶三角不等式界）：
  一次 1X 幅相测量的不确定半径
      u(z) = |z|·(ε_a + 2·sin(ε_φ/2))
  净振动由两次测量（运行 + 慢转均值）相减得到，故
      res_s = u(raw_s) + u(mean_s)
  当 |net_s| ≤ res_s 时，净振动在测量误差意义下不可分辨，
  只能标记为“不可判定”，不得当作平衡达标。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vibration import amp_phase_to_complex, complex_to_amp_phase

#: 档案至少需要录入的慢转记录条数
MIN_RECORDS = 1


@dataclass
class SensorRunout:
    """单个测点的轴跳统计。"""

    sensor: str
    mean: complex
    dispersion: float          # RMS 离散度（与振幅同单位）
    relative_dispersion: float  # 离散度/|均值|，均值为 0 时按 0 处理
    count: int


@dataclass
class RunoutSummary:
    """整份轴跳档案的逐测点统计。"""

    sensors: list[SensorRunout] = field(default_factory=list)

    def by_sensor(self) -> dict[str, SensorRunout]:
        return {s.sensor: s for s in self.sensors}

    def as_json(self) -> dict:
        out = {}
        for s in self.sensors:
            amp, phase = complex_to_amp_phase(s.mean)
            out[s.sensor] = {
                "mean": {"amplitude": amp, "phase": phase},
                "real": float(s.mean.real),
                "imag": float(s.mean.imag),
                "dispersion": float(s.dispersion),
                "relative_dispersion": float(s.relative_dispersion),
                "record_count": s.count,
            }
        return out


def _complex_uncertainty_radius(magnitude: float, amp_error: float,
                                phase_error_deg: float) -> float:
    """一次幅相测量的不确定半径（与 discrete.worst_case 同一误差模型）。"""
    return float(magnitude) * (
        float(amp_error) + 2.0 * float(np.sin(np.deg2rad(phase_error_deg) / 2.0))
    )


def build_runout_summary(
    records: list[dict],
    expected_sensors: list[str],
) -> RunoutSummary:
    """按测点聚合慢转记录为复矢量统计（不做档案有效性判定）。

    :param records: [{record_id?, speed, phase_reference,
                      measurements: [{sensor, amplitude, phase}]}]
    :param expected_sensors: 批次配置的测点名顺序
    """
    grouped: dict[str, list[complex]] = {s: [] for s in expected_sensors}
    for rec in records:
        for m in rec["measurements"]:
            if m["sensor"] in grouped:
                grouped[m["sensor"]].append(
                    amp_phase_to_complex(m["amplitude"], m["phase"])
                )

    sensors: list[SensorRunout] = []
    for name in expected_sensors:
        zs = np.array(grouped[name], dtype=complex)
        if zs.size == 0:
            continue
        mean = complex(np.mean(zs))
        dispersion = float(np.sqrt(np.mean(np.abs(zs - mean) ** 2)))
        mag = abs(mean)
        rel = float(dispersion / mag) if mag > 1e-12 else 0.0
        sensors.append(
            SensorRunout(
                sensor=name,
                mean=mean,
                dispersion=dispersion,
                relative_dispersion=rel,
                count=int(zs.size),
            )
        )
    return RunoutSummary(sensors=sensors)


@dataclass
class SensorCompensation:
    sensor: str
    raw_amplitude: float
    raw_phase: float
    compensation_amplitude: float
    compensation_phase: float
    net_amplitude: float
    net_phase: float
    resolution: float
    undecidable: bool


def compensate_vectors(
    raw: np.ndarray,
    runout: np.ndarray,
    sensor_names: list[str],
    amp_error: float,
    phase_error_deg: float,
) -> list[SensorCompensation]:
    """逐测点 z_net = z_raw - runout，并按幅相误差给出可分辨范围判定。"""
    raw = np.asarray(raw, dtype=complex)
    runout = np.asarray(runout, dtype=complex)
    net = raw - runout
    out: list[SensorCompensation] = []
    for i, s in enumerate(sensor_names):
        r_amp, r_phase = complex_to_amp_phase(raw[i])
        c_amp, c_phase = complex_to_amp_phase(runout[i])
        n_amp, n_phase = complex_to_amp_phase(net[i])
        resolution = (
            _complex_uncertainty_radius(r_amp, amp_error, phase_error_deg)
            + _complex_uncertainty_radius(c_amp, amp_error, phase_error_deg)
        )
        out.append(
            SensorCompensation(
                sensor=s,
                raw_amplitude=r_amp,
                raw_phase=r_phase,
                compensation_amplitude=c_amp,
                compensation_phase=c_phase,
                net_amplitude=n_amp,
                net_phase=n_phase,
                resolution=float(resolution),
                undecidable=bool(n_amp <= resolution),
            )
        )
    return out


def prediction_resolution(
    predicted_magnitude: float,
    amp_error: float,
    phase_error_deg: float,
) -> float:
    """方案预测净残振的可分辨范围（基于净基线的一次测量误差界）。

    安装角公差等最差情形界见 ``discrete.worst_case_residual``；
    此处仅用于“净残振是否可分辨”的判定。
    """
    return _complex_uncertainty_radius(
        predicted_magnitude, amp_error, phase_error_deg
    )
