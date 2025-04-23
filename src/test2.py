import numpy as np
from utils.common import register_gripper
import os
import robosuite as suite
import os
import yaml

from robosuite.wrappers import GymWrapper
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
from my_environments import Ultrasound, BasicEnv


register_env(BasicEnv)
register_env(Ultrasound)
register_gripper(UltrasoundProbeGripper)


# env_id = "BasicEnv"
env_id = "Ultrasound"

env_options = {}
env_options["robots"] = "Tendon"

env_options["gripper_types"] = "UltrasoundProbeGripper"
# env_options["controller_configs"] = {
#     "type": "OSC_POSE",
#     "input_max": 1,
#     "input_min": -1,
#     "output_max": [1, 1, 1, 1, 1, 1],
#     "output_min": [0, 0, 0, 0, 0, 0],
#     "kp": 300,
#     "damping_ratio": 1,
#     "impedance_mode": "fixed",
#     "kp_limits": [0, 500],
#     "kp_input_max": 1,
#     "kp_input_min": 0,
#     "damping_ratio_limits": [0, 2],
#     "position_limits": None,
#     "orientation_limits": None,
#     "uncouple_pos_ori": True,
#     "control_delta": True,
#     "interpolation": None,
#     "ramp_ratio": 0.2
# }
# env_options["control_freq"] = 500
env_options["has_renderer"] = True
env_options["has_offscreen_renderer"] = False
env_options["render_camera"] = None
env_options["use_camera_obs"] = False
env_options["use_object_obs"] = False
# env_options["horizon"] = 1000
env_options["render_gpu_device_id"] = 0
env_options["ignore_done"] = True

env = suite.make(env_id, **env_options)

obs = env.reset()
print(env._eef_xpos)

# xml_path = "../robosuite/robosuite/models/assets/robots/tendon/robot.xml"
# env = BasicEnv(xml_path)

while True:
    action = [100, 0, 0, 0, 0, 0]
    print(env.sim.data.qpos)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        print('done')
