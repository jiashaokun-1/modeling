from typing import Dict, Type, Union

from zrt.simulator import OpSimulator
from zrt.hardware import HardwareSpec
from zrt.ir import OpNode
from zrt.simulator import SimResult
from zrt.policy_model.policy_register import POLICY_MAP, PolicyType
from zrt.policy_model.policy_base_model import PolicyBaseModel

class PolicyModelManager:

    def __init__(self):
        self.target_model_map: Dict[Union[PolicyType, str], Type[PolicyBaseModel]] = {}
        self._register_model()
        self.policy_models_map: Dict[Union[PolicyType, str], PolicyBaseModel] = {}
        self._initialize_policy_model()
    
    def simulate(self, node: "OpNode", hw: "HardwareSpec", cost_model_policy: "PolicyType") -> SimResult:
        if cost_model_policy not in self.policy_models_map:
            raise ValueError(f"Policy model {cost_model_policy} not initialized")
        return self.policy_models_map[cost_model_policy].predict(node, hw)
    
    def _register_model(self):
        self.target_model_map = POLICY_MAP
        
    def _initialize_policy_model(self):
        for policy_type, model_type in self.target_model_map.items():
            self.policy_models_map[policy_type] = model_type()


    def register_backend(self, backend: OpSimulator):
        for model in self.policy_models_map.values():
            model.register_backend(backend)
