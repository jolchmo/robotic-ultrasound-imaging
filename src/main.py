import numpy as np
import time

import mujoco
import mujoco.viewer

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from my_environments import BasicEnv, visual_workspace

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

import matplotlib.pyplot as plt
import argparse


def make_env(xml):
    def _thunk():
        env = BasicEnv(xml)
        env = Monitor(env)
        env.seed(42)
        return env
    return _thunk


if __name__ == '__main__':
    # env = BasicEnv("../robosuite/robosuite/models/assets/robots/tendon/robot.xml")
    # reachable_points = env.sample_workspace()
    # np.save("waypoint.npy", reachable_points)
    # exit(0)
    # 创建解析器
    parser = argparse.ArgumentParser(description="A program with a train flag.")
    parser.add_argument('-tr', action='store_true', default=False, help='Enable training mode')
    parser.add_argument('-l', type=str, default=None, help="load model name")
    parser.add_argument('-s', type=str, default=None, help="save model name")
    parser.add_argument('-ts', type=lambda x: int(float(x)), default=1e6, help="total timestep")
    parser.add_argument('-lr', type=float, default=3e-4, help='learning rate')
    # 解析参数
    args = parser.parse_args()

    # assert 后面的东西要为True
    if args.tr and not args.l:
        print('train without loading previous model?')
        if input('confirm(y/n)').lower() == 'y':
            pass
        else:
            exit(0)
    assert not (args.tr == True and args.s == None), "Training model,please set the name of saving model"
    assert not (args.tr == False and args.l == None), "Testing model,please set the name of loading model"

    xml_path = "../robosuite/robosuite/models/assets/robots/tendon/robot.xml"
    model_policy = "PPO"

    IsTraining = args.tr
    IsTesting = not IsTraining
    IsRender = not IsTraining

    if args.l:
        IsLoadModel = True
        load_model_name = args.l
    else:
        IsLoadModel = False

    save_model_name = args.s

    print(f"------------------Training---------------------") if IsTraining else print(f"---------------------Testing:-------------------")

    if IsTraining:
        num_cpu = 16
        env = SubprocVecEnv([make_env(xml_path) for i in range(num_cpu)])

    if IsTesting:
        env = BasicEnv(xml_path)

    if IsLoadModel or IsTesting:
        try:
            model = PPO.load(f"./weights/{model_policy}/{load_model_name}", env=env, verbose=1)
            # 打印当前学习率
        except Exception as e:
            Exception(f"Error loading model: {e}")
    else:
        policy_kwargs = dict(net_arch=dict(
            pi=[64, 64],  # actor 网络的隐藏层结构
            vf=[128, 128]  # critic 网络的隐藏层结构
        ))
        model = PPO(
            policy="MlpPolicy",  # 策略類型，可以是 "MlpPolicy"、"CnnPolicy" 等
            policy_kwargs=policy_kwargs,  # 策略的網絡結構
            env=env,              # 你的環境
            learning_rate=args.lr,  # 學習率，默認為 3e-4
            n_steps=2048,         # n_steps 决定了收集多少经验
            # episode(一次完整的start->done)
            # 而iteration 通常指一次策略更新，则是一个采集了2048个episode的批次进行更新
            # 一个 iteration 对应于收集 n_steps 步经验并进行 n_epochs 次更新。
            batch_size=64,       # 每次更新的批次大小，默認為 64
            n_epochs=10,          # 每次更新時訓練的 epoch 數，默認為 10
            gamma=0.99,           # 折扣因子，默認為 0.99
            gae_lambda=0.95,      # GAE (Generalized Advantage Estimation) 的 lambda 參數，默認為 0.95
            clip_range=0.2,       # PPO 的 clip 範圍，控制策略更新的幅度，默認為 0.2
            ent_coef=0.0,         # 熵正則化係數，鼓勵探索，默認為 0.0
            vf_coef=0.5,          # 值函數損失的係數，默認為 0.5
            max_grad_norm=0.5,    # 梯度裁剪的最大範數，默認為 0.5
            verbose=1,            # 訓練過程的輸出詳細程度，0 表示無輸出，1 表示有進度條和資訊
            device="auto",        # 設備，默認為 "auto"（自動選擇 CPU 或 GPU）
            tensorboard_log="./ppo_tensorboard"  # TensorBoard 記錄目錄，用於可視化訓練過程
        )

    obs = env.reset()

    if IsTraining == True:
        # 训练过程中捕获中断
        try:
            model.learn(total_timesteps=args.ts)
            # 保存模型
            print(f"Model saved as {save_model_name}.")
            model.save(f"./weights/{model_policy}/{save_model_name}")
        except KeyboardInterrupt:
            print("\nTraining interrupted by user. Saving model...")
            if input("Do you want to save the model? (y/n): ").lower() == 'y':
                # 保存模型
                print(f"Model saved as {save_model_name}.")
                model.save(f"./weights/{model_policy}/{save_model_name}")

    if IsTesting == True:
        cnt = {"REACH": 0, "TIMEOUT": 0}
        difficult_points = []
        env.IsRender = False
        while True:
            action, _ = model.predict(obs)
            # print(action)
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                action = np.clip(action, 0, 10)  # 限制动作范围
            obs, reward, done, info = env.step(action)
            if done:
                print(f"done: {done}", f"reward: {reward}", f"info: {info}")
                print('-'*20+'reset env'+'-'*20)
                obs = env.reset()
                cnt[info['state']] += 1
                if info['state'] == "TIMEOUT":
                    difficult_points.append(info['waypoints'])
                if cnt["REACH"] + cnt["TIMEOUT"] > 500:
                    break
        print(f"reach: {cnt['REACH']}, timeout: {cnt['TIMEOUT']}")

        # print(difficult_points)
        difficult_points_set = set(tuple(point) for point in difficult_points)
        print(f"unique difficult points: {len(difficult_points_set)}")
        # np.save("difficult_points.npy", np.array(list(difficult_points_set)))

'''工作空间与可视化'''
# reachable_points = env.sample_workspace()
# np.save("waypoint.npy", reachable_points)
# reachable_points = np.load("waypoint.npy")
# env.visual_workspace(reachable_points)

# 计算每个点与 [0,0,0.5] 的欧几里得距离
# target_point = np.array([0, 0, 0.5])
# distances = np.sqrt(np.sum((reachable_points - target_point) ** 2, axis=1))

# # 找到最大距离
# max_distance = np.max(distances)

# print("最大距离:", max_distance)
# exit(0)


# PPO Log 參考
"""
-----------------------------------------
| rollout/                |             |
|    ep_len_mean          | 2.5e+03     |
|    ep_rew_mean          | 800         | 按我的獎勵設置，這個理論最大值為 1 * time_steps(2048) =2048  
| time/                   |             |
|    fps                  | 1283        |
|    iterations           | 57          |
|    time_elapsed         | 90          |
|    total_timesteps      | 116736      |
| train/                  |             |
|    approx_kl            | 0.009181175 | KL 散度（Kullback-Leibler divergence）表明策略更新幅度(較小時說明策略變化不大，可能策略還在緩慢探索或收斂。)
|    clip_fraction        | 0.112       |
|    clip_range           | 0.2         |
|    entropy_loss         | -8.37       | 熵損失值較高（負值越大，熵越大），說明策略的隨機性仍然較高，探索行為較多。
|    explained_variance   | 0.998       | 解釋方差，表明對真實回報的預測能力(越大說明模型能夠較好地預測回報)
|    learning_rate        | 0.0003      |
|    loss                 | -0.0224     |
|    n_updates            | 560         |
|    policy_gradient_loss | -0.0106     |
|    std                  | 0.977       |
|    value_loss           | 0.00534     |
-----------------------------------------
"""
