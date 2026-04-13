# 实现TensorBase类，供算子使用

class TensorBase:
    def __init__(self, shape, dtype="fp16"):
        self.shape = shape
        self.dtype = dtype
    
    def get_shape(self):
        # 返回张量的形状
        return self.shape.copy()
    
    def get_string(self):
        # 返回张量的字符串表示
        return f"Tensor(shape={self.shape}, dtype={self.dtype})"
    
    def get_flops(self):
        # 简单计算FLOPS，实际应用中可能需要更复杂的计算
        flops = 1
        for dim in self.shape:
            flops *= dim
        return flops
