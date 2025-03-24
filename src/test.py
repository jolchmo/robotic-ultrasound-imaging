import os
import robosuite as suite
import os
import yaml

from robosuite.wrappers import GymWrapper
from robosuite import load_controller_config
from robosuite.environments.base import register_env

from stable_baselines3 import PPO
from stable_baselines3.common.save_util import save_to_zip_file, load_from_zip_file
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from typing import Callable

from my_models.grippers import UltrasoundProbeGripper
from my_environments import Ultrasound
from utils.common import register_gripper

import gym


def make_robosuite_env(env_id, options, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param options: (dict) additional arguments to pass to the specific environment class initializer
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        register_gripper(UltrasoundProbeGripper)
        env = GymWrapper(suite.make(env_id, **options))
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def make_gym_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id, reward_type="dense")
        env = gym.wrappers.FlattenObservation(env)
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


if __name__ == '__main__':
    register_env(Ultrasound)

    with open("rl_config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Environment specifications
    env_options = config["robosuite"]
    env_id = env_options.pop("env_id")

    # Settings for stable-baselines RL algorithm
    sb_config = config["sb_config"]
    training_timesteps = sb_config["total_timesteps"]
    check_pt_interval = sb_config["check_pt_interval"]
    num_cpu = sb_config["num_cpu"]

    # Settings for stable-baselines policy
    policy_kwargs = config["sb_policy"]
    policy_type = policy_kwargs.pop("type")

    # Settings used for file handling and logging (save/load destination etc)
    file_handling = config["file_handling"]

    tb_log_folder = file_handling["tb_log_folder"]
    tb_log_name = file_handling["tb_log_name"]

    save_model_folder = file_handling["save_model_folder"]
    save_model_filename = file_handling["save_model_filename"]
    load_model_folder = file_handling["load_model_folder"]
    # load_model_filename = file_handling["load_model_filename"]

    load_model_filename = "tracking"

    continue_training_model_folder = file_handling["continue_training_model_folder"]
    continue_training_model_filename = file_handling["continue_training_model_filename"]

    # Join paths
    save_model_path = os.path.join(save_model_folder, save_model_filename)
    save_vecnormalize_path = os.path.join(
        save_model_folder, 'vec_normalize_' + save_model_filename + '.pkl')
    load_model_path = os.path.join(load_model_folder, load_model_filename)
    load_vecnormalize_path = os.path.join(
        load_model_folder, 'vec_normalize_' + load_model_filename + '.pkl')

    seed = config["seed"]

    # Create evaluation environment
    env_options['has_renderer'] = True
    register_gripper(UltrasoundProbeGripper)
    env_gym = GymWrapper(suite.make(env_id, **env_options))
    env = DummyVecEnv([lambda: env_gym])

    # Load normalized env
    env = VecNormalize.load(load_vecnormalize_path, env)

    # Turn of updates and reward normalization
    env.training = False
    env.norm_reward = False

    # Load model
    model = PPO.load(load_model_path, env)

    # Simulate environment
    obs = env.reset()
    eprew = 0
    step = 0
    while True:
        action, _states = model.predict(obs)
        # print(f"action: {action}")
        obs, reward, done, info = env.step(action)
        # print(action)
        # print(f'reward: {reward}')
        eprew += reward
        env_gym.render()
        step += 1
        if done:
            print(f'after {step} get total rew: {eprew}')

            # print(f'eprew: {eprew}')
            # obs = env.reset()
            eprew = 0
            step = 0

    env.close()
