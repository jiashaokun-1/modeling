
from abc import ABC
from typing import Dict, Type, Union, List
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase
from zrt.runtime_config import RuntimeConfig
from zrt.policy_model.policy_register import POLICY_MAP, PolicyType
from zrt.policy_model.policy_base_model import PolicyBaseModel

class PolicyModelManager(ABC):
    def __init__(self, rt_config: RuntimeConfig):
        self.rt_config = rt_config
        self.target_model_map: Dict[Union[PolicyType, str], Type[PolicyBaseModel]] = {}
        self._register_model()
        self.policy_models_map: Dict[Union[PolicyType, str], PolicyBaseModel] = {}
        self._initialize_policy_model()
    
    def predict(self, policy_type: Union[PolicyType, str], op: OperatorBase, input_tensor: List[TensorBase], **kwargs) -> float:
        if policy_type not in self.policy_models_map:
            raise ValueError(f"Policy model {policy_type} not initialized")
        return self.policy_models_map[policy_type].predict(op, input_tensor, **kwargs)
    
    def _register_model(self):
        self.target_model_map = POLICY_MAP
        
    def _initialize_policy_model(self):
        for policy_type, model_type in self.target_model_map.items():
            self.policy_models_map[policy_type] = model_type(self.rt_config)
