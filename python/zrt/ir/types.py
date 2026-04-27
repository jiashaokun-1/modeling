"""IR primitive types: DType enum and TensorMeta."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence


# ─────────────────────────────────────────────────────────────────────────────
# DType
# ─────────────────────────────────────────────────────────────────────────────

class DType(str, Enum):
    """Unified dtype enum, platform-agnostic."""
    FP32     = "fp32"
    FP16     = "fp16"
    BF16     = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    INT8     = "int8"
    INT4     = "int4"
    INT32    = "int32"
    INT64    = "int64"
    UINT8    = "uint8"
    BOOL     = "bool"
    UNKNOWN  = "unknown"

    @property
    def itemsize(self) -> float:
        """Bytes per element (INT4 returns 0.5)."""
        _SIZES: dict[str, float] = {
            "fp32": 4.0, "fp16": 2.0, "bf16": 2.0,
            "fp8_e4m3": 1.0, "fp8_e5m2": 1.0,
            "int8": 1.0, "int4": 0.5,
            "int32": 4.0, "int64": 8.0,
            "uint8": 1.0, "bool": 1.0, "unknown": 2.0,
        }
        return _SIZES.get(self.value, 2.0)

    @property
    def bits(self) -> float:
        return self.itemsize * 8


# ── torch dtype string → DType ────────────────────────────────────────────────

_TORCH_TO_DTYPE: dict[str, DType] = {
    "torch.float32":  DType.FP32,
    "torch.float":    DType.FP32,
    "torch.float16":  DType.FP16,
    "torch.half":     DType.FP16,
    "torch.bfloat16": DType.BF16,
    "torch.float8_e4m3fn":  DType.FP8_E4M3,
    "torch.float8_e5m2":    DType.FP8_E5M2,
    "torch.int8":   DType.INT8,
    "torch.int32":  DType.INT32,
    "torch.int":    DType.INT32,
    "torch.int64":  DType.INT64,
    "torch.long":   DType.INT64,
    "torch.uint8":  DType.UINT8,
    "torch.bool":   DType.BOOL,
}


def dtype_from_torch(s: str) -> DType:
    """Map a torch dtype string (e.g. 'torch.bfloat16') to DType."""
    return _TORCH_TO_DTYPE.get(s.strip(), DType.UNKNOWN)


def dtype_from_str(s: str) -> DType:
    """Map a DType value string (e.g. 'bf16') or torch dtype string to DType.

    Tries the DType enum value first, then falls back to the torch dtype map.
    """
    s = s.strip()
    try:
        return DType(s)
    except ValueError:
        return dtype_from_torch(s)


# ─────────────────────────────────────────────────────────────────────────────
# Shape parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_shape(s: str) -> tuple[int, ...]:
    """Parse a single shape string such as '[1, 128, 7168]' → (1, 128, 7168).

    Returns an empty tuple for empty/scalar inputs.
    """
    s = s.strip().strip("[]")
    if not s:
        return ()
    try:
        return tuple(int(x.strip()) for x in s.split(","))
    except ValueError:
        return ()


def split_shape_list(s: str) -> list[str]:
    """Split '[1, 128], [7168]' → ['[1, 128]', '[7168]'].

    Handles nested brackets correctly.
    """
    if not s:
        return []
    result: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "[":
            depth += 1
            current.append(ch)
        elif ch == "]":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                result.append(token)
            current = []
        else:
            current.append(ch)
    token = "".join(current).strip()
    if token:
        result.append(token)
    return result


def memory_bytes(shape: Sequence[int], dtype: DType) -> int:
    """Total memory in bytes for a tensor of the given shape and dtype."""
    if not shape:
        return int(math.ceil(dtype.itemsize))
    n = 1
    for d in shape:
        n *= d
    return int(math.ceil(n * dtype.itemsize))


# ─────────────────────────────────────────────────────────────────────────────
# TensorMeta
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TensorMeta:
    """Lightweight, immutable descriptor for a single tensor.

    Does not hold real data — only shape/dtype metadata.
    """
    id: str                    # stable string ID, e.g. "t42" or "op3_out0"
    shape: tuple[int, ...]
    dtype: DType
    mem_bytes: int             # pre-computed: product(shape) * dtype.itemsize

    @classmethod
    def from_shape_dtype(cls, tensor_id: str,
                         shape: tuple[int, ...], dtype: DType) -> "TensorMeta":
        return cls(
            id=tensor_id,
            shape=shape,
            dtype=dtype,
            mem_bytes=memory_bytes(shape, dtype),
        )

    @classmethod
    def from_strings(cls, tensor_id: str,
                     shape_str: str, dtype_str: str) -> "TensorMeta":
        """Construct from the string representations stored in op records."""
        shape = parse_shape(shape_str)
        dtype = dtype_from_torch(dtype_str) if dtype_str else DType.UNKNOWN
        return cls.from_shape_dtype(tensor_id, shape, dtype)

    def with_shape(self, new_shape: tuple[int, ...]) -> "TensorMeta":
        return TensorMeta.from_shape_dtype(self.id, new_shape, self.dtype)

    def with_dtype(self, new_dtype: DType) -> "TensorMeta":
        return TensorMeta.from_shape_dtype(self.id, self.shape, new_dtype)

    def __repr__(self) -> str:
        shape_str = "×".join(str(d) for d in self.shape) if self.shape else "scalar"
        return f"TensorMeta({self.id}, {shape_str}, {self.dtype.value})"
