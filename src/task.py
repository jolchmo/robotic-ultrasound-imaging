import argparse
import numpy as np
import time

import mujoco
import mujoco.viewer

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from my_environments import BasicEnv

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

from my_environments import SingleEnv


def make_env():
    def _thunk():
        env = BasicEnv()
        env = Monitor(env)
        env.seed(42)
        return env
    return _thunk


if __name__ == '__main__':

    center = np.array([0.1, 0, 1.470])

    radius = 0.1
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)  # endpoint=False 避免首尾点重复

    # 计算相对于圆心的 x 和 y 偏移量
    r_cos_theta = radius * np.cos(theta)
    r_sin_theta = radius * np.sin(theta)

    y = center[1] + r_cos_theta
    z = center[2] + r_sin_theta
    x = np.full(100, center[0])

    circle_points = np.column_stack((x, y, z))

    # # # start = init_pos+[0, -0.1, -0.01]
    # # # end = init_pos+[0,  0.1, -0.01]
    # # start = init_pos+[0, 0, -0.1]
    # # end = init_pos+[0, 0, 0.1]
    # # # # # plist = np.load("ws_1.npy")
    # # # # # start = plist[np.random.randint(0, len(plist))]
    # # # # # end = plist[np.random.randint(0, len(plist))]

    # # waypoint_line = np.linspace(start, end, num=10)
    np.save("waypoint_circle.npy", circle_points)

    env = SingleEnv()

    try:
        model_policy = "PPO"
        model_name = "test4"
        model = PPO.load(f"./weights/{model_policy}/{model_name}", env=env, verbose=1)
        # 打印当前学习率
    except Exception as e:
        Exception(f"Error loading model: {e}")

    env.step([0, 0, 0, 0, 0, 0])
    print(env.ee_pos)

    obs = env.reset()

    # env.IsRender = True
    ee_pos_list = []
    while True:

        action, _ = model.predict(obs)
        if np.any(np.isnan(action)) or np.any(np.isinf(action)):
            action = np.clip(action, 0, 20)  # 限制动作范围

        obs, reward, done, info = env.step(action)
        # print('checkpoint:', info['checkpoint'])
        print(info)
        ee_pos_list.append(env.ee_pos.copy())
        if done:
            print(f"done: {done}", f"reward: {reward}", f"info: {info}")
            print('-'*20+'reset env'+'-'*20)
            obs = env.reset()
            break
    # print(f"ee_pos_list: {ee_pos_list}")
    print(f"len(ee_pos_list): {len(ee_pos_list)}")
    np.save("circle.npy", ee_pos_list)
