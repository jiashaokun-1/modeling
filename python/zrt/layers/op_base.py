from dataclasses import dataclass
from typing import List, Type

# 定义DType枚举
class DType:
    FP16 = "fp16"
    INT8 = "int8"

@dataclass
class SimulateResult:
    cost: float
    latency: float

class OperatorBase:
    def __init__(self, op_model, op_name, is_vector, op_dtype):
        self.model = op_model
        self.name = op_name
        self.is_vector = is_vector
        self.op_dtype = op_dtype
        self.inputs = []
        self.outputs = []
        self.compute_formula = ""
        self.compute_flops = 0
        
    def set_memory_bytes(self):
        # 简单实现，计算输入和输出的内存大小
        self.memory_bytes = 0
        for tensor in self.inputs:
            if hasattr(tensor, 'get_shape') and hasattr(tensor, 'dtype'):
                shape = tensor.get_shape()
                dtype_size = 2 if self.op_dtype == DType.FP16 else 1  # 假设FP16是2字节，INT8是1字节
                size = 1
                for dim in shape:
                    size *= dim
                self.memory_bytes += size * dtype_size
        for tensor in self.outputs:
            if hasattr(tensor, 'get_shape') and hasattr(tensor, 'dtype'):
                shape = tensor.get_shape()
                dtype_size = 2 if self.op_dtype == DType.FP16 else 1
                size = 1
                for dim in shape:
                    size *= dim
                self.memory_bytes += size * dtype_size

class OpVectorBase(OperatorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, op_name, is_vector=True, op_dtype=DType.FP16)
        self.static_cost = 2

class OpCubeBase(OperatorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, op_name, is_vector=False, op_dtype=DType.INT8)
        self.static_cost = 5

class OpMixBase(OperatorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, op_name, is_vector=False, op_dtype=DType.FP16)
        self.static_cost = 3

class OpCommBase(OperatorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, op_name, is_vector=False, op_dtype=DType.FP16)
        self.static_cost = 10

OP_CLASS_REGISTRY = {}

def op_register(names: str | List[str]):
    def decorator(cls: Type[OperatorBase]):
        if isinstance(names, str):
            op_names = [names]
        elif isinstance(names, list) and all(isinstance(name, str) for name in names):
            op_names = names
        else:
            raise ValueError("names must be a string or a list of strings")
        for name in op_names:
            OP_CLASS_REGISTRY[name] = cls
        return cls
    return decorator

def get_class_by_name(name: str) -> type[OperatorBase]:
    
    if name not in OP_CLASS_REGISTRY:
        raise ValueError(f"Operator class not found for name: {name}")
    return OP_CLASS_REGISTRY[name]
    