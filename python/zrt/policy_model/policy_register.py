
from typing import Dict, Type
from enum import Enum
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .policy_base_model import PolicyBaseModel

class PolicyType(Enum):
    PRIORITY = 'priority'
    OOTB_PERFORMANCE = 'ootb_performance'
    OPERATOR_OPTIMIZATION = 'operator_optimization'
    SYSTEM_DESIGN = 'system_design'

POLICY_MAP: Dict[PolicyType, Type['PolicyBaseModel']] = {}
def register_model():
    from .priority_model import PriorityModel
    from .open_box_model import OpenBoxModel
    from .op_aptimize_model import OperatorOptimizationModel
    from .micro_architecture_model import SystemDesignModel


    POLICY_MAP.update({
        PolicyType.PRIORITY: PriorityModel,
        PolicyType.OOTB_PERFORMANCE: OpenBoxModel,
        PolicyType.OPERATOR_OPTIMIZATION: OperatorOptimizationModel,
        PolicyType.SYSTEM_DESIGN: SystemDesignModel,
    })
