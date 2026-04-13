from zrt.layers.op_base import OpVectorBase, OpCubeBase, op_register
from zrt.tensor_base import TensorBase
from zrt.input_param import InputParam

from typing import List

@op_register("RMSNorm")
class RMSNorm(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "RMSNorm")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # RMSNorm的输出形状与输入相同
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
            if len(in_shape) == 2:
                in_shape[0] = b * s
            elif len(in_shape) == 3:
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("GemmaRMSNorm")
class GemmaRMSNorm(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "GemmaRMSNorm")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # GemmaRMSNorm的输出形状与输入相同
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
            if len(in_shape) == 2:
                in_shape[0] = b * s
            elif len(in_shape) == 3:
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("RMSNormGated")
class RMSNormGated(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "RMSNormGated")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # RMSNormGated的输出形状与输入相同
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
            if len(in_shape) == 2:
                in_shape[0] = b * s
            elif len(in_shape) == 3:
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("RopeKernel")
class RopeKernel(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "RopeKernel")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # RopeKernel的输出形状与输入相同
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

@op_register("MoEGatingTopk")
class MoEGatingTopk(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "MoEGatingTopk")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # MoEGatingTopk的输出形状取决于输入形状和topk值
            input_shape = input_tensor[0].get_shape()
            topk = kwargs.get('topk', 2)  # 默认topk值
            output_shape = input_shape.copy()
            output_shape[-1] = topk
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

@op_register("RopeInterleave")
class RopeInterleave(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "RopeInterleave")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # RopeInterleave的输出形状与输入相同
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
