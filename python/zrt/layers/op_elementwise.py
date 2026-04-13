from typing import List

from zrt.layers.op_base import OpVectorBase, OpCubeBase, op_register
from zrt.tensor_base import TensorBase
from zrt.input_param import InputParam

@op_register(["Add", "AddInplace"])
class Add(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Add")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        if len(input_tensor) > 1:
            y_tensor = input_tensor[1]
            self.inputs.append(y_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.get_shape(), dtype=self.op_dtype))

        self.compute_formula = x_tensor.get_string()
        self.compute_flops = x_tensor.get_flops()

        self.set_memory_bytes()

        return self.outputs[0]

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = input_tensor[0].get_shape()
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)

        if len(input_tensor) == 1:
            return [in0]
        
        in_shape_y = input_tensor[1].get_shape()
        if len(in_shape_y) == 2:
            in_shape_y[0] = b * s
        else:
            in_shape_y[0] = b
            in_shape_y[1] = s

        in1 = TensorBase(shape=in_shape_y, dtype=input_tensor[1].dtype)
        return [in0, in1]

@op_register(["Mul", "MulInplace"])
class Mul(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Mul")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        if len(input_tensor) > 1:
            y_tensor = input_tensor[1]
            self.inputs.append(y_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.get_shape(), dtype=self.op_dtype))

        self.compute_formula = x_tensor.get_string()
        self.compute_flops = x_tensor.get_flops()

        self.set_memory_bytes()

        return self.outputs[0]

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = input_tensor[0].get_shape()
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)

        if len(input_tensor) == 1:
            return [in0]
        
        in_shape_y = input_tensor[1].get_shape()
        if len(in_shape_y) == 2:
            in_shape_y[0] = b * s
        else:
            in_shape_y[0] = b
            in_shape_y[1] = s

        in1 = TensorBase(shape=in_shape_y, dtype=input_tensor[1].dtype)
        return [in0, in1]

@op_register("Softmax")
class Softmax(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Softmax")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.get_shape(), dtype=self.op_dtype))

        self.compute_formula = x_tensor.get_string()
        self.compute_flops = x_tensor.get_flops()

        self.set_memory_bytes()

        return self.outputs[0]

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = input_tensor[0].get_shape()
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Sin")
class Sin(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Sin")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.get_shape(), dtype=self.op_dtype))

        self.compute_formula = x_tensor.get_string()
        self.compute_flops = x_tensor.get_flops()

        self.set_memory_bytes()

        return self.outputs[0]

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = input_tensor[0].get_shape()
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Cos")
class Cos(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Cos")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.get_shape(), dtype=self.op_dtype))

        self.compute_formula = x_tensor.get_string()
        self.compute_flops = x_tensor.get_flops()

        self.set_memory_bytes()

        return self.outputs[0]

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = input_tensor[0].get_shape()
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Histc")
class Histc(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Histc")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        # Histc的输出形状取决于bins参数
        bins = kwargs.get('bins', 10)
        self.outputs.append(TensorBase(shape=[bins], dtype=self.op_dtype))

        self.compute_formula = x_tensor.get_string()
        self.compute_flops = x_tensor.get_flops()

        self.set_memory_bytes()

        return self.outputs[0]

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = input_tensor[0].get_shape()
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Sort")
class Sort(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Sort")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.get_shape(), dtype=self.op_dtype))

        self.compute_formula = x_tensor.get_string()
        self.compute_flops = x_tensor.get_flops()

        self.set_memory_bytes()

        return self.outputs[0]

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = input_tensor[0].get_shape()
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Sum")
class Sum(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Sum")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        # Sum的输出形状取决于dim参数
        dim = kwargs.get('dim', None)
        if dim is None:
            # 全局求和，输出标量
            self.outputs.append(TensorBase(shape=[], dtype=self.op_dtype))
        else:
            # 沿指定维度求和，输出形状为原形状去掉该维度
            output_shape = x_tensor.get_shape().copy()
            if dim < len(output_shape):
                output_shape.pop(dim)
            self.outputs.append(TensorBase(shape=output_shape, dtype=self.op_dtype))

        self.compute_formula = x_tensor.get_string()
        self.compute_flops = x_tensor.get_flops()

        self.set_memory_bytes()

        return self.outputs[0]

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = input_tensor[0].get_shape()
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Cumsum")
class Cumsum(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "Cumsum")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.get_shape(), dtype=self.op_dtype))

        self.compute_formula = x_tensor.get_string()
        self.compute_flops = x_tensor.get_flops()

        self.set_memory_bytes()

        return self.outputs[0]

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = input_tensor[0].get_shape()
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]
