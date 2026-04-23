from zrt.hardware import HardwareSpec
from zrt.ir import OpNode
from zrt.simulator import OpSimulator, SimResult


class LookupSimulator(OpSimulator):
    name = "lookup"
    priority = 1
    def __init__(self):
        super().__init__()

    def can_simulate(self, node: "OpNode", hw: "HardwareSpec") -> bool:
        return False

    def simulate(self, node: "OpNode", hw: "HardwareSpec") -> SimResult:
        pass


