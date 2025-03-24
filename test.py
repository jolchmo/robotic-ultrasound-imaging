import random
import time
import numpy as np
import mujoco_py
import robosuite as suite
import os


# 创建 Tendon 机器人模型实例
# 假设你的 Tendon 类位于 robosuite.models.robots 模块中
robot_model = suite.models.robots.Tendon()

# 获取 MJCF 模型 (返回的是 lxml.etree._Element 对象)
mjcf_model = robot_model.get_model(mode="mujoco_py")


sim = mujoco_py.MjSim(mjcf_model)

# 创建查看器
viewer = mujoco_py.MjViewer(sim)


# 初始化关节目标位置
target_qpos = np.zeros(6)

print(sim.data.qpos)

# 运行仿真循环
times = 0
while True:

    # 将目标位置应用到仿真
    sim.data.qpos[random.sample(range(0, 5), 1)[0]] = random.sample([-0.1, 0.1], 1)[0]

    # 执行一步仿真
    sim.step()

    # 渲染场景
    viewer.render()
