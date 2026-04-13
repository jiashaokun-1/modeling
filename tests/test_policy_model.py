#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试policy_model模块的功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from zrt.policy_model.policy_register import register_model, PolicyType
from zrt.policy_model.policy_model_manager import PolicyModelManager

# 假设的RuntimeConfig类
class RuntimeConfig:
    def __init__(self):
        self.ai_chip_config = None

# 假设的OperatorBase类
class OperatorBase:
    pass

# 假设的TensorBase类
class TensorBase:
    pass

def test_policy_model():
    """测试policy_model模块的功能"""
    print("开始测试policy_model模块...")
    
    # 注册模型
    register_model()
    print("模型注册成功")
    
    # 创建RuntimeConfig实例
    rt_config = RuntimeConfig()
    
    # 创建PolicyModelManager实例
    try:
        manager = PolicyModelManager(rt_config)
        print("PolicyModelManager创建成功")
    except Exception as e:
        print(f"PolicyModelManager创建失败: {e}")
        return False
    
    # 测试预测功能
    op = OperatorBase()
    input_tensor = [TensorBase()]
    
    # 测试所有策略类型
    policy_types = [
        PolicyType.PRIORITY,
        PolicyType.OOTB_PERFORMANCE,
        PolicyType.OPERATOR_OPTIMIZATION,
        PolicyType.SYSTEM_DESIGN
    ]
    
    for policy_type in policy_types:
        try:
            result = manager.predict(policy_type, op, input_tensor)
            print(f"测试{policy_type.value}成功，结果: {result}")
        except Exception as e:
            print(f"测试{policy_type.value}失败: {e}")
            return False
    
    print("所有测试通过！")
    return True

if __name__ == "__main__":
    test_policy_model()
