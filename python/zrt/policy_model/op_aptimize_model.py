from python.zrt.hardware import HardwareSpec
from python.zrt.ir import OpNode
from python.zrt.simulator import SimResult
from .policy_base_model import PolicyBaseModel

class OperatorOptimizationModel(PolicyBaseModel):
    def __init__(self):
        super().__init__()
    
    def predict(self, node: "OpNode", hw: "HardwareSpec") -> SimResult:
        pass
