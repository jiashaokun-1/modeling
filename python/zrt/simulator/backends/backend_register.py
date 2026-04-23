from enum import Enum, auto
from typing import Dict, Type

from zrt.simulator import OpSimulator
import logging
logger = logging.getLogger(__name__)

class BackendType(Enum):
    ROOFLINE = auto()
    LOOKUP = auto()
    TILESIM = auto()

BACKEND_MAP: Dict[BackendType, Type['OpSimulator']] = {}

def register_backend():
    from zrt.simulator.backends.roofline import RooflineSimulator
    from zrt.simulator.backends.lookup import LookupSimulator
    from zrt.simulator.backends.tilesim import TilesimSimulator

    BACKEND_MAP.update({
        BackendType.ROOFLINE: RooflineSimulator,
        BackendType.LOOKUP: LookupSimulator,
        BackendType.TILESIM: TilesimSimulator
    })
