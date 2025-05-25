import argparse
import numpy as np
import time

import mujoco
import mujoco.viewer

from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from my_environments import SingleEnv

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback


def make_env():
    def _thunk():
        env = SingleEnv()
        env = Monitor(env)
        env.seed(42)
        return env
    return _thunk


def get_args():
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
    return args


if __name__ == '__main__':
    args = get_args()

    model_policy = "PPO"
    IsTraining = True if args.tr else False
    IsTesting = False if args.tr else True
    IsRender = False if args.tr else True
    env_options = {}

    print(f"------------------Training---------------------") if IsTraining else print(f"---------------------Testing:-------------------")

    # 如果训练，需要多线程，按需要设置
    if IsTraining:
        num_cpu = 8
        env = SubprocVecEnv([make_env(**env_options) for i in range(num_cpu)])
    else:
        env = SingleEnv(**env_options)

    if args.l:
        try:
            print(f"Loading model from {args.l}")
            print(f"./weights/{model_policy}/{args.l}")
            model = PPO.load(f"./weights/{model_policy}/{args.l}", env=env, verbose=1)
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
    if IsTraining:
        # 训练过程中捕获中断，方便中间打断也可以保存
        try:
            model.learn(total_timesteps=args.ts)
            model.save(f"./weights/{model_policy}/{args.s}")
        except KeyboardInterrupt:
            print("Training interrupted by user. Saving model...")
            if input("Do you want to save the model? (y/n): ").lower() == 'y':
                model.save(f"./weights/{model_policy}/{args.s}")
        except Exception as e:
            print(f"Error during training: {e}")
            print(f"Model saved as {args.s}.")

    else:
        cnt = {"REACH": 0, "TIMEOUT": 0, "reward": [], "distance": [], "TO_distance": []}
        env.IsRender = True
        while True:
            action, _ = model.predict(obs)
            if np.any(np.isnan(action)) or np.any(np.isinf(action)):
                action = np.clip(action, 0, 20)  # 限制动作范围

            obs, reward, done, info = env.step(action)
            if done:
                print(f"done: {done}", f"reward: {reward}", f"info: {info}")
                print('-'*20+'reset env'+'-'*20)
                obs = env.reset()
                cnt[info['state']] += 1
                cnt["reward"].append(reward)
                cnt["distance"].append(info['distance'])
                if info['state'] == "TIMEOUT":
                    cnt["TO_distance"].append(info['distance'])
                if cnt["REACH"] + cnt["TIMEOUT"] > 500:
                    break
        print('在经过500次的测试后，模型的表现如下：')
        print(f"reach: {cnt['REACH']}, timeout: {cnt['TIMEOUT']}")
        print(f"reward: {round(np.mean(cnt['reward']),2)}, distance: {round(np.mean(cnt['distance']),2)}cm")
        print(f"超时的测试，平均距离为: {round(np.mean(cnt['TO_distance']),2)}cm")
