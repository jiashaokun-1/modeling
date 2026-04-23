from zrt.hardware import HardwareSpec
from zrt.ir import OpNode
from zrt.simulator import OpSimulator, SimResult

class TilesimSimulator(OpSimulator):
    name = "tilesim"
    priority = 2

    def can_simulate(self, node: "OpNode", hw: "HardwareSpec") -> bool:
        return False

    def simulate(self, node: "OpNode", hw: "HardwareSpec") -> SimResult:
        pass