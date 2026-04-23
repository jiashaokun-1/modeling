"""SimulatorHub: routes simulation requests to the best available backend.

Usage::

    from python.zrt.simulator import SimulatorHub
    from python.zrt.simulator.backends import RooflineSimulator
    from python.zrt.hardware import load as load_hw

    hub = SimulatorHub()
    hub.register(RooflineSimulator())

    hw = load_hw("ascend_910b")
    result = hub.simulate(node, hw)
    results = hub.simulate_graph(graph, hw)
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .base import OpSimulator
from .cache import SimCache
from .result import SimResult
from zrt.policy_model.policy_model_manager import PolicyModelManager
from zrt.policy_model.policy_register import PolicyType

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.node import OpNode
    from python.zrt.hardware.spec import HardwareSpec

logger = logging.getLogger(__name__)


class SimulatorHub:
    """Route operator simulation to the highest-priority capable backend.

    Backends are sorted by ``priority`` (descending).  The first backend
    whose ``can_simulate()`` returns True is used.  Results are cached by
    content hash so repeated calls with the same (op, hw) pair are free.

    A ``RooflineSimulator`` is always registered at priority 0 as the
    universal fallback.
    """

    def __init__(self, *, with_roofline: bool = True) -> None:
        self._backends: list[OpSimulator] = []
        self._cache = SimCache()

        self.policy_model_manager = PolicyModelManager()

    # ── registration ─────────────────────────────────────────────────────────

    def register(self, backend: OpSimulator) -> None:
        """Register a backend and re-sort by priority (highest first)."""
        self.policy_model_manager.register_backend(backend)
        logger.debug(
            "SimulatorHub: registered backend %r (priority=%d)",
            backend.name, backend.priority,
        )

    # ── simulation ───────────────────────────────────────────────────────────

    def simulate(self, node: "OpNode", hw: "HardwareSpec", cost_model_policy: "PolicyType") -> SimResult:
        """Simulate a single node; uses cache when available."""
        cached = self._cache.get(node, hw)
        if cached is not None:
            return cached

        result = self.policy_model_manager.simulate(node, hw, cost_model_policy)

        self._cache.put(node, hw, result)
        if result is None:
            raise RuntimeError(
                f"No backend can simulate op_type={node.op_type!r} on hw={hw.name!r}. "
                "Register a RooflineSimulator or pass with_roofline=True."
            )
        return result


    def simulate_graph(
        self,
        graph: "OpGraph",
        hw: "HardwareSpec",
        cost_model_policy: "PolicyType" = PolicyType.PRIORITY
    ) -> dict[str, SimResult]:
        """Simulate every node in *graph* in topological order.

        Returns a dict mapping node_id → SimResult.
        """
        results: dict[str, SimResult] = {}
        nodes = graph.topo_sort()
        logger.debug(
            "SimulatorHub.simulate_graph: %d nodes, hw=%s, costmodel_policy=%s",
            len(nodes), hw.name, cost_model_policy.name
        )
        for node in nodes:
            results[node.id] = self.simulate(node, hw, cost_model_policy)
        return results

    # ── cache control ────────────────────────────────────────────────────────

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "hits":   self._cache.hits,
            "misses": self._cache.misses,
            "size":   len(self._cache),
        }

    # ── convenience factory ──────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "SimulatorHub":
        """Return a hub with the Roofline backend pre-registered."""
        return cls(with_roofline=True)
