from typing import List
from enum import Enum, auto
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase
from zrt.runtime_config import RuntimeConfig
from zrt.policy_model.policy_base_model import PolicyBaseModel

class CostModelManager:
    def __init__(self, ai_chip_config, rt_config, model_target):
        pass
    
    def predict(self, op, input_tensor, **kwargs):
        return 0.0

class Modeltype(Enum):
    LOOKUP = auto()

    TILESIM_ENGI = auto()
    TILESIM_THEO = auto()
    TILESIM_ENGI_DSL = auto()

    THEO_MODEL = auto()

class PriorityModel(PolicyBaseModel):
    def __init__(self, rt_config: RuntimeConfig):
        super().__init__(rt_config)
        self.model_target = [Modeltype.LOOKUP, Modeltype.TILESIM_THEO, Modeltype.THEO_MODEL]
        self.costmodel_manager = CostModelManager(rt_config.ai_chip_config, self.rt_config, self.model_target)

    def predict(self, op: OperatorBase, input_tensor: List[TensorBase], **kwargs):
        return self.costmodel_manager.predict(op, input_tensor, **kwargs)

