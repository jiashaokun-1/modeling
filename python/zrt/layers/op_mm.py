from zrt.layers.op_base import OpVectorBase, OpCubeBase, op_register
from zrt.tensor_base import TensorBase
from zrt.input_param import InputParam

from typing import List

@op_register("Bmm")
class Bmm(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Bmm")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if len(input_tensor) >= 2:
            # Bmm的输出形状取决于输入形状
            # 假设输入是 [batch, m, k] 和 [batch, k, n]
            # 输出是 [batch, m, n]
            input_shape1 = input_tensor[0].get_shape()
            input_shape2 = input_tensor[1].get_shape()
            output_shape = [input_shape1[0], input_shape1[1], input_shape2[2]]
            self.outputs.append(TensorBase(shape=output_shape, dtype=self.op_dtype))

        # 假设存在这些方法
        if input_tensor:
            self.compute_formula = input_tensor[0].get_string()
            self.compute_flops = input_tensor[0].get_flops()

        # 假设存在这个方法
        self.set_memory_bytes()

        return self.outputs[0] if self.outputs else None

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = tensor.get_shape()
            if len(in_shape) == 3:
                in_shape[0] = b
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("MatMul")
class MatMul(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "MatMul")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if len(input_tensor) >= 2:
            # MatMul的输出形状取决于输入形状
            # 假设输入是 [m, k] 和 [k, n]
            # 输出是 [m, n]
            input_shape1 = input_tensor[0].get_shape()
            input_shape2 = input_tensor[1].get_shape()
            output_shape = [input_shape1[0], input_shape2[1]]
            self.outputs.append(TensorBase(shape=output_shape, dtype=self.op_dtype))

        # 假设存在这些方法
        if input_tensor:
            self.compute_formula = input_tensor[0].get_string()
            self.compute_flops = input_tensor[0].get_flops()

        # 假设存在这个方法
        self.set_memory_bytes()

        return self.outputs[0] if self.outputs else None

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = tensor.get_shape()
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("GroupedMatMul")
class GroupedMatMul(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "GroupedMatMul")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if len(input_tensor) >= 2:
            # GroupedMatMul的输出形状取决于输入形状
            input_shape1 = input_tensor[0].get_shape()
            input_shape2 = input_tensor[1].get_shape()
            output_shape = input_shape1.copy()
            output_shape[-1] = input_shape2[-1]
            self.outputs.append(TensorBase(shape=output_shape, dtype=self.op_dtype))

        # 假设存在这些方法
        if input_tensor:
            self.compute_formula = input_tensor[0].get_string()
            self.compute_flops = input_tensor[0].get_flops()

        # 假设存在这个方法
        self.set_memory_bytes()

        return self.outputs[0] if self.outputs else None

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = tensor.get_shape()
            if len(in_shape) == 3:
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("Mm")
class Mm(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Mm")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if len(input_tensor) >= 2:
            # Mm的输出形状取决于输入形状
            input_shape1 = input_tensor[0].get_shape()
            input_shape2 = input_tensor[1].get_shape()
            output_shape = [input_shape1[0], input_shape2[1]]
            self.outputs.append(TensorBase(shape=output_shape, dtype=self.op_dtype))

        # 假设存在这些方法
        if input_tensor:
            self.compute_formula = input_tensor[0].get_string()
            self.compute_flops = input_tensor[0].get_flops()

        # 假设存在这个方法
        self.set_memory_bytes()

        return self.outputs[0] if self.outputs else None

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = tensor.get_shape()
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("Linear")
class Linear(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Linear")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # Linear的输出形状取决于输入形状和输出维度
            input_shape = input_tensor[0].get_shape()
            output_dim = kwargs.get('output_dim', 768)  # 默认输出维度
            output_shape = input_shape.copy()
            output_shape[-1] = output_dim
            self.outputs.append(TensorBase(shape=output_shape, dtype=self.op_dtype))

        # 假设存在这些方法
        if input_tensor:
            self.compute_formula = input_tensor[0].get_string()
            self.compute_flops = input_tensor[0].get_flops()

        # 假设存在这个方法
        self.set_memory_bytes()

        return self.outputs[0] if self.outputs else None

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = tensor.get_shape()
            if len(in_shape) == 2:
                in_shape[0] = b * s
            elif len(in_shape) == 3:
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("Einsum")
class Einsum(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Einsum")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # Einsum的输出形状取决于爱因斯坦求和约定
            # 这里简化处理，使用第一个输入的形状
            output_shape = input_tensor[0].get_shape()
            self.outputs.append(TensorBase(shape=output_shape, dtype=self.op_dtype))

        # 假设存在这些方法
        if input_tensor:
            self.compute_formula = input_tensor[0].get_string()
            self.compute_flops = input_tensor[0].get_flops()

        # 假设存在这个方法
        self.set_memory_bytes()

        return self.outputs[0] if self.outputs else None

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = tensor.get_shape()
            if len(in_shape) == 3:
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("ColumnParallelLinear")
class ColumnParallelLinear(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "ColumnParallelLinear")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # ColumnParallelLinear的输出形状取决于输入形状和输出维度
            input_shape = input_tensor[0].get_shape()
            output_dim = kwargs.get('output_dim', 768)  # 默认输出维度
            output_shape = input_shape.copy()
            output_shape[-1] = output_dim
            self.outputs.append(TensorBase(shape=output_shape, dtype=self.op_dtype))

        # 假设存在这些方法
        if input_tensor:
            self.compute_formula = input_tensor[0].get_string()
            self.compute_flops = input_tensor[0].get_flops()

        # 假设存在这个方法
        self.set_memory_bytes()

        return self.outputs[0] if self.outputs else None

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = tensor.get_shape()
            if len(in_shape) == 2:
                in_shape[0] = b * s
            elif len(in_shape) == 3:
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs
