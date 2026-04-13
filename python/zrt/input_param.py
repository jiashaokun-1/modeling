# 实现InputParam类，供算子的build_dynamic_input方法使用

class InputParam:
    def __init__(self, batch_size=32, seq_len=1024):
        self.batch_size = batch_size
        self.seq_len = seq_len
