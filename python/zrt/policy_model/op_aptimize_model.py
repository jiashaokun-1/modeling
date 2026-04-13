
from typing import List
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase
from zrt.runtime_config import RuntimeConfig
from zrt.policy_model.policy_base_model import PolicyBaseModel

class OperatorOptimizationModel(PolicyBaseModel):
    def __init__(self, rt_config: RuntimeConfig):
        super().__init__(rt_config)
    
    def predict(self, op: OperatorBase, input_tensor: List[TensorBase], **kwargs) -> float:
        return 0.0
