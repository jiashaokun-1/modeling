from python.zrt.hardware import HardwareSpec
from python.zrt.ir import OpNode
from python.zrt.simulator import SimResult
from .policy_base_model import PolicyBaseModel

class PriorityModel(PolicyBaseModel):
    def __init__(self):
        super().__init__()


    def predict(self, node: "OpNode", hw: "HardwareSpec") -> SimResult:
        for backend in self._backends:
            if backend.can_simulate(node, hw):
                result = backend.simulate(node, hw)
                return result
        return None


