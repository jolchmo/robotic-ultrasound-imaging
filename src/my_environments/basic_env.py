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
            z_offset=0.0,
        )

        return self.torso

    def robot_init(self):
        if self.robot_id != "Tendon" or self.gripper_type != "UltrasoundProbeGripper":
            raise ValueError("robot_id and gripper_type must be set to 'Tendon' and 'ultrasound_probe' respectively.")
        self.idn = 0
        self.pf = f"robot{str(self.idn)}"
        self.robot = Tendon(self.idn)
        self.gripper = gripper_factory(self.gripper_type, self.idn)
        self.arena = UltrasoundArena()
        self.torso = self.toroso_init()

        # get pos and orientation of robot
        self.robot_offset = np.array([-0.39, 0, self.table_full_size[0]+0.67])  # -90
        # self.robot_offset = np.array([0., 0, self.table_full_size[0]+0.9])  # 0
        # self.robot_offset = np.array([-0.7, 0, self.table_full_size[0]+0.75])  # 45

        self.robot_quat = np.array([0.70711, 0.0, 0.70711, 0.0])
        # self.robot_quat = np.array([1, 0, 0, 0])
        # self.robot_quat = np.array([0.81915, 0.0, 0.57358, 0.0])

        # init robot pos and orientation
        self.robot.set_base_xpos(self.robot_offset)
        self.robot.set_base_quat(convert_quat(self.robot_quat, to="wxyz"))

        # attach gripper to robot
        self.robot.add_gripper(self.gripper)
        self.robot.merge(self.arena)
        self.robot.merge(self.torso)
        xml_string = ET.tostring(self.robot.root, encoding='unicode')
        # with open('robot.xml', 'w') as f:
        #     f.write(xml_string)
#
        return mujoco.MjModel.from_xml_string(xml_string)

    def get_actuator_id(self):
        for i in range(1, 7):
            name = f"{self.pf}_joint_{str(i)}"
            joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self._ref_joint_pos_indexes.append(joint_id)

    def set_init_qpos(self, init_qpos):
        if not self._ref_joint_pos_indexes:
            self.get_actuator_id()
            print
        self.data.qpos[self._ref_joint_pos_indexes] = init_qpos

    def __init__(self):

        # self.transform_home_to_work = get_transform_matrix(self.robot_offset, self.robot_quat)
        # self.transform_work_to_home = np.linalg.inv(self.transform_home_to_work)

        # init robot attach gripper to robot
        self.robot_id = "Tendon"
        self.gripper_type = "UltrasoundProbeGripper"

        # 定义状态空间和动作空间
        # 这里不要用self.model.nq因为还有别的joint
        self.n_joints = 6
        self.n_actuators = 6
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=20.0, shape=(self.n_actuators,), dtype=np.float32)

        self.table_full_size = np.array([0.8, 0.8, 0.05])
        self.placement_initializer = None
        self._ref_joint_pos_indexes = []

        # self.robot = self.robot_init()
        self.mj_model = self.robot_init()
        # robot_xml = 'robot.xml'
        # self.mj_model = mujoco.MjModel.from_xml_path(robot_xml)
        self.data = mujoco.MjData(self.mj_model)

        # init_qpos = [0.0246, 0.0246, 0.0331, 0.161, 0.15, 0.0448]
        # self.set_init_qpos(init_qpos)

        # 影响着采样的数量，
        self.timestep = self.mj_model.opt.timestep
        self.contrl_hz = 50
        self.control_timestep = 1 / self.contrl_hz  # 则policy step一次，仿真 control/timestep(10)次

        # reward 相关
        # 一个ep 的时间限制和最大奖励
        self.ep_time = 60.0
        self.ep_price = self.ep_time * self.contrl_hz
        self.z_forces_mean = 0.0  # init
        self.goal_contact_z_force = 5.0
        self.goal_xquat = np.array([1.0, 0.0, 0.0, 0.0])
        # reward权重
        # position 权重
        self.pos_error_mul = 5.0
        self.pos_reward_mul = 0.6

        # orientation 权重
        self.xquat_error_mul = 4.0
        self.xquat_reward_mul = 0.3

        # force 权重
        self.force_error_mul = 0.1
        self.force_reward_mul = 0.1  # 0.7

        # reward惩罚权重
        self.time_penalty_mul = -0.1

        # vec reward
        self.ee_accel_penalty_mul = -0.05
        self.prev_ee_vel = None
        self.prev_ee_pos = None
        # action reward
        self.action_rate_penalty_mul = -0.1
        self.prev_action = None

        # reward terminal condition
        self.require_stable = True
        self.stable_second = 0.1
        self.stable_frame = self.stable_second / self.control_timestep
        self.distance_treshold = 1e-2
        self.distance_end = 1e-2

        # 渲染模式
        self.viewer = None
        self.render_context = None
        self.IsRender = False

        self.cnt = 0
        # 路点
        self.checkpoint = 0  # 现在在哪个点
        self.IsWaypointRandom = False
        # self.rate = 1
        self.waypoint_array = np.load("liver_loop_path.npy")
        # self.waypoint_array = np.load("liver_path.npy")
        # self.waypoint_array = self.waypoint_array[:int(self.rate/10*len(self.waypoint_array))]
        self.waypoint_geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "robot0_waypoint")

        # torso
        self.torso_site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "body_top_site")
        self.body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "body")
        self.probe_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "gripper0_gripper_base")
        # 初始化状态
        self.reset()

    def get_waypoint(self):
        # self.waypoint = self.data.site_xpos[self.torso_site_id]
        # waypoint = [-0.06063048, -0.03755068, 1.07151569]
        # if self.IsWaypointRandom:
        #     waypoint = self.waypoint_array[np.random.randint(0, len(self.waypoint_array))]
        waypoint = self.waypoint_array[self.checkpoint]
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
        self.set_random_torso()
        self.waypoint_pos = self.get_waypoint()
        self.mj_model.site_pos[self.waypoint_geom_id] = self.waypoint_pos
        self.cnt = 0
        self.checkpoint = 0  # 现在在哪个点
        return self._get_obs()

    def step(self, action):
        self.z_forces = []

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

        diff_vec = self.ee_pos - self.waypoint_pos
        dx = diff_vec[0]
        dy = diff_vec[1]
        dz = diff_vec[2]
        z_tolerance = 0.03  # 假设单位是米
        if abs(dz) <= z_tolerance:
            dz_error_contribution = 0.0
        else:
            dz_error_contribution = np.sign(dz) * (abs(dz) - z_tolerance)
        pos_error = np.linalg.norm([dx, dy, dz_error_contribution])
        pos_reward = self.pos_reward_mul * np.exp(- self.pos_error_mul * pos_error)

        xquat_error = self.xquat_error_mul * \
            distance_quat(self.ee_xquat, self.goal_xquat)
        xquat_reward = self.xquat_reward_mul * np.exp(-xquat_error)

        force_error = self.force_error_mul * \
            abs(self.z_forces_mean - self.goal_contact_z_force)
        force_reward = self.force_reward_mul * np.exp(-force_error)

        time_penalty = self.time_penalty_mul * (self.data.time / self.ep_time)
        # smooth action
        ee_accel_penalty = 0.0
        action_rate_penalty = 0.0
        if self.prev_ee_pos is not None and self.prev_ee_vel is not None and hasattr(self, 'dt'):
            current_ee_vel = (self.ee_pos - self.prev_ee_pos) / self.control_timestep
            ee_accel = (current_ee_vel - self.prev_ee_vel) / self.control_timestep
            # Using squared magnitude of acceleration for penalty
            ee_accel_penalty = self.ee_accel_penalty_mul * np.sum(np.square(ee_accel))
        else:  # First step or dt not defined
            current_ee_vel = np.zeros_like(self.ee_pos)

        # 2. Control Action Rate Penalty
        # if self.prev_action is not None:
        #     action_diff = action - self.prev_action
        #     # Using squared magnitude of action difference
        #     action_rate_penalty = self.action_rate_penalty_mul * np.sum(np.square(action_diff))

        reward = pos_reward + xquat_reward + force_reward + time_penalty\
            + ee_accel_penalty

        # 计数器.计算ep中是否连续到达目标点,超出则重置

        self.cnt = self.cnt + 1 if pos_error < self.distance_treshold else 0
        # print(self.cnt)

        if (self.cnt > self.stable_frame and pos_error < self.distance_end) or (not self.require_stable and self.cnt == 1):
            reward = self.ep_price * 2 / len(self.waypoint_array)
            state = "REACH"
            self.checkpoint += 1
            self.waypoint_pos = self.get_waypoint()
            self.mj_model.site_pos[self.waypoint_geom_id] = self.waypoint_pos
        if self.data.time > self.ep_time:
            state = "TIMEOUT"
            done = True
        if self.checkpoint == len(self.waypoint_array) - 1:
            done = True

        # save previous state for next step
        self.prev_ee_pos = self.ee_pos.copy()
        self.prev_ee_vel = current_ee_vel.copy()
        self.prev_action = action.copy()

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

        # print(f"eef_pos: {self.ee_pos}")
        # print(f"waypoint_pos: {self.waypoint_pos}")

        obs = np.concatenate((self.ee_pos, self.waypoint_pos, self.ee_xquat, self.goal_xquat.copy(), np.array([self.z_forces_mean])))
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
    def ee_pos(self):
        ee_pos = self.data.site_xpos[self.ee_id]
        return ee_pos

    # ----------------------------
    @property
    def ee_xmat(self):
        return np.array(self.data.site_xmat[self.ee_id]).reshape(3, 3)

    @property
    def ee_xquat(self):
        """
        Returns:
            np.array: (x,y,z,w) End Effector
        """
        return mat2quat(self.ee_xmat)

    # ----------------------------

    @property
    def joint_pos(self):
        return self.data.qpos

    # ----------------------------
    @property
    def eef_contact_force(self):
        return self.data.cfrc_ext[self.probe_id][:]
