from abc import ABC, abstractmethod

from python.zrt.simulator import OpSimulator
from python.zrt.simulator.backends.backend_register import BackendType, BACKEND_MAP
from python.zrt.hardware import HardwareSpec
from python.zrt.ir import OpNode
from python.zrt.simulator import SimResult


class PolicyBaseModel(ABC):
    def __init__(self):
        self.backend_target = [BackendType.LOOKUP, BackendType.TILESIM, BackendType.ROOFLINE]
        self._backends: list[OpSimulator] = []
        self._register_backends()

    @abstractmethod
    def predict(self, node: "OpNode", hw: "HardwareSpec") -> SimResult:
        pass

    def _register_backends(self):
        for backend_type in self.backend_target:
            backend = BACKEND_MAP[backend_type]()
            self._backends.append(backend)
            self._backends.sort(key=lambda b: b.priority, reverse=True)

    def register_backend(self, backend):
        self._backends.append(backend)
        self._backends.sort(key=lambda b: b.priority, reverse=True)
