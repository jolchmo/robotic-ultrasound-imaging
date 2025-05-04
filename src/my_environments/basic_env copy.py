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


import os
os.environ['MUJOCO_GL'] = 'egl'  # 设置渲染模式为 EGL


class BasicEnv(gym.Env):
    def __init__(self, robot_xml):
        super(BasicEnv, self).__init__()
        # 加载 MuJoCo 模型
        self.model = mujoco.MjModel.from_xml_path(robot_xml)
        # self.gripper = mujoco.MjModel.from_xml_path("./my_models/assets/grippers/ultrasound_probe_gripper.xml")

        # self.attach_gripper()
        self.data = mujoco.MjData(self.model)
        # 影响着采样的数量，
        self.timestep = self.model.opt.timestep
        self.contrl_hz = 50
        self.control_timestep = 1 / self.contrl_hz  # 则policy step一次，仿真 control/step(25)次
        # 定义状态空间和动作空间
        # 这里不要用self.model.nq因为还有别的joint
        self.n_joints = 6
        self.n_actuators = self.model.nu  # 执行器数量（动作空间维度）
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.action_space = spaces.Box(low=0, high=10.0, shape=(self.n_actuators,), dtype=np.float32)

        # 渲染模式
        self.viewer = None
        self.render_context = None
        self.IsRender = False

        # 路点
        # self.waypoint_pos = [-0.04, -0.12, 0.47]  # p2 done
        self.IsWaypointRandom = True
        self.waypoint_array = np.load("waypoint.npy")
        # self.waypoint_array = np.load("difficult_points.npy")
        # self.rate = 10
        # self.waypoint_array = self.waypoint_array[:int(self.rate/10*len(self.waypoint_array))]
        self.waypoint_geom_id = -1
        try:
            self.waypoint_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "waypoint")
        except:
            Exception("Warning: 'waypoint' site not found in the model. Using default position.")

        # 初始化状态
        self.reset()

    # def attach_gripper(self):
    #     arm_name = self.ee_name
    #     if arm_name in self.model.gripper:
    #         raise ValueError("Attempts to add multiple grippers to one body")

    #     self.model.merge(self.gripper, merge_body=arm_name)
    #     self.grippers[arm_name] = self.gripper

    def get_waypoint(self):
        if self.IsWaypointRandom:
            return self.waypoint_array[np.random.randint(0, len(self.waypoint_array))]
        return [-0.04, -0.12, 0.47]

    def set_waypoint(self, waypoint):
        """
        设置路点位置
        :param waypoint: 路点位置
        """
        self.waypoint_pos = waypoint
        self.model.site_pos[self.waypoint_geom_id] = self.waypoint_pos

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.waypoint_pos = self.get_waypoint()
        self.model.site_pos[self.waypoint_geom_id] = self.waypoint_pos
        return self._get_obs()

    def step(self, action):
        # action
        for i in range(int(self.control_timestep/self.timestep)):
            self.data.ctrl[:] = action
            mujoco.mj_step(self.model, self.data)
            if self.IsRender:
                self.render()

        # observation
        obs = self._get_obs()
        done = False

        time_limit = 10.0

        # error_mul = 5
        error_mul = 6.0
        pos_reward_mul = 1.0
        # print(self.ee_pos[:] - self.waypoint_pos[:])
        ee_pos = self.ee_pos[:]
        distance = np.linalg.norm((ee_pos - self.waypoint_pos[:]))
        reward = pos_reward_mul * np.exp(-error_mul * distance)
        # reward -= 0.1 * self.data.time / time_limit
        all_price = time_limit / self.control_timestep
        state = "RUNNING"
        if np.linalg.norm((ee_pos - self.waypoint_pos[:])) < 1e-2:
            # 如果到达目标点，半程(前面给全程奖励，模型对失败的情况不敏感)
            reward = all_price
            state = "REACH"
            done = True
        if self.data.time > time_limit:
            # 如果超时，给个负奖励,除了没有奖励外，给个负奖励,要不然在附近点和到达目标点的奖励差不多，模型不去到目标点。
            state = "TIMEOUT"
            done = True
        info = {"distance": f"{round(distance*100,2)}cm", 'state': state, "waypoints": self.waypoint_pos, "time": self.data.time}
        return obs, reward, done, info

    # 單次的渲染
    def render(self, mode="human"):
        if mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
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
                self.render_context = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150.value)

            if not hasattr(self, 'scene') or self.scene is None:
                self.scene = mujoco.MjvScene(self.model, maxgeom=10000)

            if not hasattr(self, 'camera') or self.camera is None:
                self.camera = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(self.camera)

            # 更新场景
            mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), None, self.camera, mujoco.mjtCatBit.mjCAT_ALL.value, self.scene)

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
        return "ee"

    @property
    def ee_id(self):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_name)

    @property
    def ee_pos(self):
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

    @property
    def eef_contact_force(self):
        return self.data.cfrc_ext[self.probe_id][:]


def sample_workspace(env, num_samples=20000, max_time_per_sample=0.02):
    """
    随机采样关节角度，记录末端执行器的位置，探索机器人工作空间。

    Args:
        num_samples (int): 采样点数量，默认值为 10000。
        max_time_per_sample (float): 每个采样点仿真的最大时间

    Returns:
        np.ndarray: 末端执行器位置的数组，形状为 (num_samples, 3)。
    """
    import time
    # 预分配数组以提高效率
    reachable_points = np.zeros((num_samples, 3))  # 假设末端执行器位置是 3D 坐标
    num_joints = env.n_joints  # 获取关节数量
    for i in range(num_samples):
        # 随机生成所有关节的角度，范围在关节限制内
        random_joints = np.random.uniform(0, 1, size=(num_joints))
        env.data.ctrl[:] = random_joints  # 设置所有关节的控制输入

        start_time = time.time()
        while time.time() - start_time < max_time_per_sample:
            # 执行一步仿真
            mujoco.mj_step(env.model, env.data)
            # 如果系统稳定，可以提前退出（可选：添加稳定性判断条件）
            # 例如：检查速度是否接近零
            if np.all(np.abs(env.data.qvel[:6]) < 1e-3):  # 假设 qvel 是关节速度
                break

        # 记录末端执行器位置
        reachable_points[i] = env.ee_pos.copy()  # 假设 ee_pos 是末端执行器位置

        # 可选：渲染仿真过程（调试用）
        # self.render()

    return reachable_points


def visual_workspace(workspace_points):
    """可视化工作空间"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
