"""领域错误：携带结构化细节，由 FastAPI 异常处理器转换为 422 响应。"""

from __future__ import annotations

from typing import Any


class DomainError(Exception):
    """业务规则违反。details 中携带相关运行与约束信息。"""

    code = "domain_error"
    status_code = 422

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class NotFoundError(DomainError):
    code = "not_found"
    status_code = 404


class SpeedDeviationError(DomainError):
    """运行转速相对基准转速偏差超限。"""

    code = "speed_deviation"


class PhaseReferenceConflict(DomainError):
    """同一批次内运行的相位基准不一致。"""

    code = "phase_reference_conflict"


class InsufficientTrials(DomainError):
    """试重运行不足或试重矩阵秩亏，无法辨识全部影响系数。"""

    code = "insufficient_trials"


class IllConditionedMatrix(DomainError):
    """试重矩阵病态，影响系数不可信。"""

    code = "ill_conditioned_matrix"


class NoFeasibleCombination(DomainError):
    """在给定孔位、配重规格与质量上限约束下无可安装的离散组合。"""

    code = "no_feasible_combination"


class CalibrationMissing(DomainError):
    code = "calibration_missing"


class MixedCorrectionConfigError(DomainError):
    """加重/去料混合校正的几何或约束配置自相矛盾（如孔不可钻、限值为负）。"""

    code = "mixed_correction_config_error"


class NoFeasibleMixedCorrection(DomainError):
    """给定孔位、质量、厚度与单面改变量限制下不存在可行的混合校正候选。"""

    code = "no_feasible_mixed_correction"


class RunoutProfileInvalid(DomainError):
    """轴跳档案不满足使用条件（测点缺失、超速、离散度超限等）。"""

    code = "runout_profile_invalid"


class RunoutPhaseReferenceConflict(DomainError):
    """轴跳记录与被补偿测量的相位基准约定不一致。"""

    code = "runout_phase_reference_conflict"
