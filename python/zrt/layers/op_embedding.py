from zrt.layers.op_base import OpVectorBase, op_register
from zrt.tensor_base import TensorBase
from zrt.input_param import InputParam

from typing import List

@op_register("Embedding")
class Embedding(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Embedding")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # Embedding的输出形状通常是 [batch, seq_len, embedding_dim]
            input_shape = input_tensor[0].get_shape()
            # 假设最后一维是词索引，输出将增加embedding_dim维度
            output_shape = input_shape.copy()
            output_shape.append(kwargs.get('embedding_dim', 768))  # 默认embedding维度
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
            if len(in_shape) == 1:
                # 假设形状为 [seq_len]
                in_shape = [b * s]
            elif len(in_shape) == 2:
                # 假设形状为 [batch, seq_len]
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs
