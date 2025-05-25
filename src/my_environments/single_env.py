import time
from collections import OrderedDict
import mujoco
import mujoco.viewer
import mujoco.egl
import numpy as np

import gym
from gym import spaces
from robosuite.utils.transform_utils import mat2quat
from robosuite.models.grippers import gripper_factory
from robosuite.models.robots.manipulators.tendon_robot import Tendon
import xml.etree.ElementTree as ET

import os
os.environ['MUJOCO_GL'] = 'egl'  # 设置渲染模式为 EGL


class SingleEnv(gym.Env):
    def __init__(self):
        super(SingleEnv, self).__init__()
        # 加载 MuJoCo 模型
        # robot_xml = os.path.join(os.path.dirname(__file__), "assets", "robot.xml")
        self.robot = Tendon(0)
        xml_string = ET.tostring(self.robot.root, encoding='unicode')

        self.mj_model = mujoco.MjModel.from_xml_string(xml_string)
        with open('single.xml', 'w') as f:
            f.write(xml_string)
        # self.gripper = mujoco.MjModel.from_xml_path("./my_models/assets/grippers/ultrasound_probe_gripper.xml")

        # self.attach_gripper()
        self.data = mujoco.MjData(self.mj_model)
        # 影响着采样的数量，
        self.timestep = self.mj_model.opt.timestep
        self.contrl_hz = 50
        self.control_timestep = 1 / self.contrl_hz  # 则policy step一次，仿真 control/step(25)次
        # 定义状态空间和动作空间
        # 这里不要用self.mj_model.nq因为还有别的joint
        self.n_joints = 6
        self.n_actuators = self.mj_model.nu  # 执行器数量（动作空间维度）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=10.0, shape=(self.n_actuators,), dtype=np.float32)

        # 渲染模式
        self.viewer = None
        self.render_context = None
        self.IsRender = False

        # 路点
        self.cnt = 0
        self.checkpoint = 0
        self.IsWaypointRandom = False
        self.waypoint_array = np.load("waypoint_circle.npy")
        self.waypoint_geom_id = -1
        try:
            self.waypoint_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "robot0_waypoint")
            print(f"waypoint_geom_id: {self.waypoint_geom_id}")
        except:
            Exception("Warning: 'waypoint' site not found in the model. Using default position.")

        # 初始化状态
        self.reset()

    def get_waypoint(self):
        if self.IsWaypointRandom:
            waypoint = self.waypoint_array[np.random.randint(0, len(self.waypoint_array))]
        else:
            waypoint = self.waypoint_array[self.checkpoint]
        return waypoint

    def next_waypoint(self):
        if self.waypoint_geom_id == -1:
            raise ValueError("Waypoint site not found in the model.")
        waypoint = self.waypoint_array[self.checkpoint]
        self.set_waypoint(waypoint)

    def set_waypoint(self, waypoint):
        """
        设置路点位置
        :param waypoint: 路点位置
        """
        self.waypoint_pos = waypoint
        self.mj_model.site_pos[self.waypoint_geom_id] = self.waypoint_pos

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.data)
        self.set_waypoint(self.get_waypoint())
        return self._get_obs()

    def step(self, action):
        self.z_forces = []
        # 政策每给出一次action，环节都要执行control_timestep/self.timestep=，才会有下一次的动作
        for i in range(int(self.control_timestep/self.timestep)):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.mj_model, self.data)
            if self.IsRender:
                self.render()

        self.z_forces_mean = np.mean(self.z_forces)
        obs = self._get_obs()
        done = False
        state = "RUNNING"

        # 需要增加精度时调整这个，默认为5.0,在观察到精度不足的时候，可以修改到6，视情况修改mul，但可能会带来问题(一个ep可以得到的最大reward变了，)
        self.pos_reward_mul = 1.0
        self.pos_error_mul = 5.0
        self.distance_treshold = 2e-2
        self.stable_second = 0.05
        self.contrl_hz = 50
        self.ep_time = 30.0
        self.time_penalty_mul = -0.1
        self.stable_frame = self.stable_second / self.control_timestep
        self.ep_price = self.ep_time * self.contrl_hz
        pos_error = np.linalg.norm(self.ee_pos - self.waypoint_pos)
        pos_reward = self.pos_reward_mul * np.exp(- self.pos_error_mul * pos_error)
        time_penalty = self.time_penalty_mul * (self.data.time / self.ep_time)

        reward = pos_reward + time_penalty
        # 计数器.计算ep中是否连续到达目标点,超出则重置

        self.cnt = self.cnt + 1 if pos_error < self.distance_treshold else 0
        # print(self.cnt)

        if (self.cnt > self.stable_frame):
            reward = self.ep_price / 10
            state = "REACH"
            # done = True
            print('reach one')
            self.checkpoint += 1
            # self.set_waypoint(np.array([0.1, 0, 1.47])+[0, 0, 0.2])
            self.next_waypoint()
        if self.data.time > self.ep_time:
            state = "TIMEOUT"
            done = True
        if self.checkpoint == len(self.waypoint_array) - 1:
            reward = self.ep_price
            done = True
        info = {"distance": round(pos_error*100, 2), 'state': state, "waypoints": self.waypoint_pos,
                "time": round(self.data.time, 2), "ee_pos": self.ee_pos, "checkpoint": self.checkpoint}
        return obs, reward, done, info

    # 單次的渲染

    def render(self, mode="human"):
        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.data)
            self.viewer.sync()
        elif mode == "rgb_array":
            # 设置视口大小（可以根据需要调整）
            width, height = 640, 480
            viewport = mujoco.MjrRect(0, 0, width, height)

            # 初始化渲染相关的对象（如果还没初始化）
            if not hasattr(self, 'render_context') or self.render_context is None:
                # 创建一个 OpenGL 上下文
                gl_context = mujoco.egl.GLContext(width, height)
                gl_context.make_current()
                self.render_context = mujoco.MjrContext(self.mj_model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

            if not hasattr(self, 'scene') or self.scene is None:
                self.scene = mujoco.MjvScene(self.mj_model, maxgeom=10000)

            if not hasattr(self, 'camera') or self.camera is None:
                self.camera = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(self.camera)

            # 更新场景
            mujoco.mjv_updateScene(self.mj_model, self.data, mujoco.MjvOption(), None, self.camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)

            # 渲染场景
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            depth = np.zeros((height, width), dtype=np.float32)
            mujoco.mjr_render(viewport, self.scene, self.render_context)

            # 读取渲染结果
            mujoco.mjr_readPixels(rgb, depth, viewport, self.render_context)

            # 翻转图像（MuJoCo 渲染结果是上下颠倒的）
            rgb = np.flipud(rgb)
            return rgb

        else:
            raise ValueError("Invalid render mode. Use 'human' or 'rgb_array'.")
        return None

    def _get_obs(self):
        # 状态包含智能体位置和路点位置
        return np.concatenate([self.ee_pos[:], self._eef_xquat, self.waypoint_pos], dtype=np.float32)

    @property
    def ee_name(self):
        return "robot0_ee"

    @property
    def ee_id(self):
        ee_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name)
        if ee_id == -1:
            raise ValueError("End effector ID not found. Please check the model.")
        return ee_id

    @property
    def ee_pos(self):
        if self.ee_id == -1:
            raise ValueError("End effector ID not found. Please check the model.")
        ee_pos = self.data.site_xpos[self.ee_id]
        return ee_pos

    @property
    def _eef_xmat(self):
        return np.array(self.data.site_xmat[self.ee_id]).reshape(3, 3)

    @property
    def _eef_xquat(self):
        return mat2quat(self._eef_xmat)

    @property
    def joint_pos(self):
        return self.data.qpos
