from zrt.hardware import HardwareSpec
from zrt.ir import OpNode
from zrt.simulator import SimResult
from zrt.policy_model.policy_base_model import PolicyBaseModel

class SystemDesignModel(PolicyBaseModel):
    def __init__(self):
        super().__init__()
    
    def predict(self, node: "OpNode", hw: "HardwareSpec") -> SimResult:
        pass
