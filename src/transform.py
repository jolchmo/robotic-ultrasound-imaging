# test_basic_env_seed.py
import mujoco
import numpy as np
from my_environments import BasicEnv  # 导入你的环境类

# 创建环境实例 (传入需要的参数，例如模型路径等)
# 注意：如果 BasicEnv.__init__ 仍然依赖于 Robosuite 的 suite.make 创建模型，
# 你可能需要先注册环境，并用 suite.make 创建，而不是直接 new BasicEnv()
# 假设 suite.make 是创建入口:
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.environments.base import register_env
from my_models.grippers import UltrasoundProbeGripper
from utils.common import register_gripper

register_gripper(UltrasoundProbeGripper)  # 注册 gripper
register_env(BasicEnv)  # 注册你的 BasicEnv

# 使用 suite.make 创建环境，这通常会调用你的 BasicEnv 的 __init__
# 传入 suite.make 需要的参数，特别是 env_id 和 robots 等
# env_id 应该是你注册 BasicEnv 时的 ID
# 例如:
env = suite.make(
    "BasicEnv",  # 使用你注册的 env_id
)

# 如果需要，包装成 Gym 接口
env = GymWrapper(env)


print(f"Environment class: {type(env)}")
print(f"Does environment have seed method? {hasattr(env, 'seed')}")

# 尝试设置种子
seed_value = 42
print(f"Attempting to set seed to {seed_value}")
try:
    env.seed(seed_value)
    print(f"Successfully set seed to {seed_value}")
except AttributeError as e:
    print(f"AttributeError when setting seed: {e}")

# ... 可以尝试调用 reset ...
# obs = env.reset()
