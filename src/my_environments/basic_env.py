import os
import xml.etree.ElementTree as ET
from robosuite.models.grippers import gripper_factory
import time

import mujoco
import mujoco.viewer
import mujoco.egl
import numpy as np

import gym
from gym import spaces
from robosuite.utils.transform_utils import mat2quat
from robosuite.models.robots.manipulators.tendon_robot import Tendon
from robosuite.utils.mjcf_utils import array_to_string, ROBOT_COLLISION_COLOR, string_to_array
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.transform_utils import convert_quat, quat2mat, mat2euler, euler2mat, mat2quat
from robosuite.environments.manipulation.single_arm_env import SingleArmEnv
from utils.quaternion import distance_quat, difference_quat


from utils.common import register_gripper
from my_models.grippers import UltrasoundProbeGripper
from my_models.arenas import UltrasoundArena
from my_models.objects import SoftTorsoObject, BoxObject, SoftBoxObject, ClientBodyObject


register_gripper(UltrasoundProbeGripper)

os.environ['MUJOCO_GL'] = 'egl'  # 设置渲染模式为 EGL


def get_transform_matrix(translation_vector, quaternion):
    """
    生成包含平移和旋转的齐次变换矩阵。

    Args:
        translation_vector (list or np.ndarray): 包含 [tx, ty, tz] 的平移向量。
        quaternion (list or np.ndarray): 包含 [x, y, z, w] 的四元数。

    Returns:
        np.ndarray: 4x4 的齐次变换矩阵。
    """
    rotation_matrix = quat2mat(quaternion)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = translation_vector
    return transform_matrix


class BasicEnv(gym.Env):

    def home2work(self, x_pos):
        x = np.array(list(x_pos) + [1.0])
        return (self.transform_home_to_work @ x)[:3]

    def work2home(self, x_pos):
        x = np.array(list(x_pos) + [1.0])
        return (self.transform_work_to_home @ x)[:3]

    def toroso_init(self):
        self.torso = ClientBodyObject(name="body")
        self.torso_offset = [0, 0, 0.8]
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=[self.torso],
            x_range=[-0.0, 0.0],  # [-0.12, 0.12],
            y_range=[-0.0, 0.0],  # [-0.12, 0.12],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.torso_offset,
            z_offset=0.005,
        )

        return self.torso

    def robot_init(self):
        if self.robot_id != "Tendon" or self.gripper_type != "UltrasoundProbeGripper":
            raise ValueError("robot_id and gripper_type must be set to 'Tendon' and 'ultrasound_probe' respectively.")
        self.idn = 0
        self.robot = Tendon(self.idn)
        self.gripper = gripper_factory(self.gripper_type, self.idn)
        self.arena = UltrasoundArena()
        # self.torso = self.toroso_init()

        # get pos and orientation of robot
        self.robot_offset = self.robot.base_xpos_offset["table"](self.table_full_size[0])
        # xml requires quaternion in wxyz format
        # self.robot_quat = np.array([0.70711, 0.0, 0.70711, 0.0])
        self.robot_quat = np.array([-1, 0, 0, 0])
        # init robot pos and orientation
        self.robot.set_base_xpos(self.robot_offset)
        self.robot.set_base_quat(convert_quat(self.robot_quat, to="wxyz"))

        # attach gripper to robot
        self.robot.add_gripper(self.gripper)
        self.robot.merge(self.arena)
        # self.robot.merge(self.torso)
        return self.robot

    def __init__(self):
        # init robot attach gripper to robot
        self.robot_id = "Tendon"
        self.gripper_type = "UltrasoundProbeGripper"
        self.table_full_size = np.array([0.8, 0.8, 0.05])
        self.placement_initializer = None

        self.robot = self.robot_init()

        self.transform_home_to_work = get_transform_matrix(self.robot_offset, self.robot_quat)
        self.transform_work_to_home = np.linalg.inv(self.transform_home_to_work)

        # print(self.transform_home_to_work)

        xml_string = ET.tostring(self.robot.root, encoding='unicode')

        # with open(robot.xml", "w") as f:
        #     f.write(xml_string)

        self.mj_model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.mj_model)

        # 影响着采样的数量，
        self.timestep = self.mj_model.opt.timestep
        self.contrl_hz = 50
        self.control_timestep = 1 / self.contrl_hz  # 则policy step一次，仿真 control/timestep(10)次

        # reward 相关
        # 一个ep 的时间限制和最大奖励
        self.ep_time = 10.0
        self.ep_price = self.ep_time * self.contrl_hz
        self.z_forces_mean = 0.0  # init
        self.goal_contact_z_force = 5.0
        self.goal_xquat = np.array([1.0, 0.0, 0.0, 0.0])
        # reward权重
        # position 权重
        self.pos_error_mul = 5.0
        self.pos_reward_mul = 1.0

        # orientation 权重
        self.xquat_error_mul = 0.2
        self.xquat_reward_mul = 0.0

        # force 权重
        self.force_error_mul = 0.1
        self.force_reward_mul = 0  # 0.7

        # reward惩罚权重
        self.time_penalty_mul = -0.1
        # reward terminal condition
        self.require_stable = False
        self.stable_second = 0.5
        self.stable_frame = self.stable_second / self.control_timestep
        self.distance_treshold = 1e-2

        # 定义状态空间和动作空间
        # 这里不要用self.model.nq因为还有别的joint
        self.n_joints = 6
        self.n_actuators = self.mj_model.nu  # 执行器数量（动作空间维度）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=10.0, shape=(self.n_actuators,), dtype=np.float32)

        # 渲染模式
        self.viewer = None
        self.render_context = None
        self.IsRender = False

        self.cnt = 0
        # 路点
        self.IsWaypointRandom = False
        self.rate = 10
        self.waypoint_array = np.load("waypoint_top.npy")
        self.waypoint_array = self.waypoint_array[:int(self.rate/10*len(self.waypoint_array))]
        self.waypoint_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "robot0_waypoint")

        # torso
        self.torso_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "body_top_site")
        self.body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "body")

        # 初始化状态
        self.reset()

    def get_waypoint(self):
        # self.waypoint = self.data.site_xpos[self.torso_site_id]
        waypoint = [0.03688948, 0.0083384, 1.06451953]
        if self.IsWaypointRandom:
            waypoint = self.waypoint_array[np.random.randint(0, len(self.waypoint_array))]
        return waypoint

    def set_random_torso(self):
        object_placements = self.placement_initializer.sample()
        for obj_pos, orientation_info, obj in object_placements.values():
            target_joint_name = 'body_torso_free_joint'  # Replace with actual way to get joint name from obj

            joint_id = mujoco.mj_name2id(
                self.mj_model,
                mujoco.mjtObj.mjOBJ_JOINT,
                target_joint_name
            )

            if joint_id == -1:
                print(f"Error: Joint '{target_joint_name}' not found! Check XML.")
                # Handle error
            else:
                qpos_adr = self.mj_model.jnt_qposadr[joint_id]
                desired_orientation_quat = np.array([1.0, 0.0, 0.0, 0.0])
                joint_qpos_values = np.concatenate([np.array(obj_pos), desired_orientation_quat])
                self.data.qpos[qpos_adr: qpos_adr + 7] = joint_qpos_values
                mujoco.mj_forward(self.mj_model, self.data)

    def reset(self):
        mujoco.mj_resetData(self.mj_model, self.data)
        # self.set_random_torso()
        self.waypoint_pos = self.get_waypoint()
        self.mj_model.site_pos[self.waypoint_geom_id] = self.waypoint_pos
        self.cnt = 0
        return self._get_obs()

    def step(self, action):
        self.z_forces = []
        # 政策每给出一次action，环节都要执行control_timestep/self.timestep=，才会有下一次的动作
        for i in range(int(self.control_timestep/self.timestep)):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.mj_model, self.data)
            self.z_forces.append(self.eef_contact_force[2])
            if self.IsRender:
                self.render()

        self.z_forces_mean = np.mean(self.z_forces)
        obs = self._get_obs()
        done = False
        state = "RUNNING"

        # 需要增加精度时调整这个，默认为5.0,在观察到精度不足的时候，可以修改到6，视情况修改mul，但可能会带来问题(一个ep可以得到的最大reward变了，)

        pos_error = self.pos_error_mul * \
            np.linalg.norm((self._eef_pos - self.waypoint_pos))
        pos_reward = self.pos_reward_mul * np.exp(-pos_error)

        xquat_error = self.xquat_error_mul * \
            distance_quat(self._eef_xquat, self.goal_xquat)
        xquat_reward = self.xquat_reward_mul * np.exp(-xquat_error)

        force_error = self.force_error_mul * \
            abs(self.z_forces_mean - self.goal_contact_z_force)
        force_reward = self.force_reward_mul * np.exp(-force_error)

        time_penalty = self.time_penalty_mul * (self.data.time / self.ep_time)
        reward = pos_reward + xquat_reward + force_reward + time_penalty

        # 计数器.计算ep中是否连续到达目标点,超出则重置

        self.cnt = self.cnt + 1 if pos_error < self.distance_treshold else 0

        if self.cnt > self.stable_frame or (self.require_stable and self.cnt == 1):
            reward = self.ep_price
            state = "REACH"
            done = True
        if self.data.time > self.ep_time:
            state = "TIMEOUT"
            done = True
        info = {"distance": f"{round(pos_error*100,2)}cm", 'state': state, "waypoints": self.waypoint_pos, "time": self.data.time}
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
        """
        Constructs the observation array for the reinforcement learning agent.

        The observation includes the end-effector's current pose,
        the waypoint's pose, the difference between them,
        the current contact force, and the difference between
        the current and goal contact force.
        """
        # Print for debugging purposes (optional)

        observation_components = []

        # print(f"eef_pos: {self._eef_pos}")
        # print(f"waypoint_pos: {self.waypoint_pos}")

        distance_pos = self._eef_pos.copy() - self.waypoint_pos.copy()

        difference_xquat = difference_quat(self._eef_xquat.copy(), self.goal_xquat.copy())

        # pos_diff = self.waypoint_pos - self._eef_pos
        # xquat_diff = difference_quat(self._eef_xquat, self.goal_xquat)
        # pose_diff = np.concatenate([pos_diff, xquat_diff])
        # observation_components.append(pose_diff)

        obs = np.concatenate((distance_pos, difference_xquat, np.array([self.z_forces_mean])))

        # print(f"obs: {obs}")

        return obs

    # ----------------------------

    @property
    def ee_name(self):
        if self.gripper:
            return "gripper0_grip_site"
        return "ee"

    @property
    def ee_id(self):
        return mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name)

    @property
    def _eef_pos(self):
        _eef_pos = self.data.site_xpos[self.ee_id]
        return _eef_pos

    # ----------------------------
    @property
    def _eef_xmat(self):
        return np.array(self.data.site_xmat[self.ee_id]).reshape(3, 3)

    @property
    def _eef_xquat(self):
        """
        Returns:
            np.array: (x,y,z,w) End Effector
        """
        return mat2quat(self._eef_xmat)

    # ----------------------------

    @property
    def joint_pos(self):
        return self.data.qpos

    # ----------------------------
    @property
    def eef_contact_force(self):
        return [0, 0, 0.5]
        return self.data.cfrc_ext[self.ee_id][:]
