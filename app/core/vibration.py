"""复振动与复配重的幅相转换工具。

约定：相位单位为度，振动 z = A·exp(iφ)，配重 w = m·exp(iθ)，
角度均以键相基准的滞后（lag）正方向为准，由调用方保证同一批次内一致。
"""

from __future__ import annotations

import numpy as np


def amp_phase_to_complex(amplitude: float, phase_deg: float) -> complex:
    """振幅 + 相位(度) -> 复数。"""
    return complex(amplitude) * np.exp(1j * np.deg2rad(phase_deg))


def complex_to_amp_phase(z: complex) -> tuple[float, float]:
    """复数 -> (振幅, 相位[0,360)度)。"""
    return float(abs(z)), float(np.rad2deg(np.angle(z)) % 360.0)


def vector_to_complex(mass: float, angle_deg: float, radius: float = 1.0) -> complex:
    """质量(g) + 角度(度) (+半径) -> 复配重向量。半径为 1 时即质量向量。"""
    return complex(mass) * complex(radius) * np.exp(1j * np.deg2rad(angle_deg))


def normalize_angle(angle_deg: float) -> float:
    return float(angle_deg % 360.0)
