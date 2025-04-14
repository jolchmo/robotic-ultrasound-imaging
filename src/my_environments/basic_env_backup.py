from collections import OrderedDict
import os
import numpy as np
import pandas as pd
import re
from klampt.model import trajectory
import roboticstoolbox as rtb

from spatialmath import SE3

from robosuite.utils.transform_utils import convert_quat, quat2mat, mat2euler
from robosuite.utils.mjcf_utils import CustomMaterial

from robosuite.environments.manipulation.single_arm_env import SingleArmEnv

from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.observables import Observable, sensor
from robosuite.models.base import MujocoModel

import robosuite.utils.transform_utils as T

from my_models.objects import SoftTorsoObject, BoxObject, SoftBoxObject, ClientBodyObject
from robosuite.models.tasks.task import Task
from my_models.arenas import BasicArena
from utils.quaternion import distance_quat, difference_quat


class BasicEnv(SingleArmEnv):
    """
    Simplified ultrasound task environment for a single robot arm.
    """

    def __init__(
        self,
        robots,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        use_camera_obs=True,
        use_object_obs=True,
        reward_scale=1.0,
        reward_shaping=False,
        placement_initializer=None,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="frontview",
        render_collision_mesh=False,
        render_visual_mesh=True,
        render_gpu_device_id=-1,
        control_freq=20,
        horizon=1000,
        ignore_done=False,
        hard_reset=True,
        camera_names="agentview",
        camera_heights=256,
        camera_widths=256,
        camera_depths=False,
        early_termination=False,
        save_data=False,
        deterministic_trajectory=False,
        torso_solref_randomization=False,
        initial_probe_pos_randomization=False,
        use_box_torso=True,
    ):

        self.horizon = horizon
        self.goal_quat = np.array([0, 0, 0, 1])
        self.pos_error_mul = 90
        self.ori_error_mul = 0.2
        self.pos_reward_mul = 5
        self.ori_reward_mul = 1

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            mount_types="default",
            gripper_types=gripper_types,
            initialization_noise=None,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
        )

    def reward(self, action=None):
        """
        Reward function for the task.
        """
        ee_current_ori = convert_quat(self._eef_xquat, to="wxyz")
        ee_desired_ori = convert_quat(self.goal_quat, to="wxyz")

        # Orientation reward
        ori_error = self.ori_error_mul * np.linalg.norm(ee_current_ori - ee_desired_ori)
        ori_reward = self.ori_reward_mul * np.exp(-ori_error)

        return ori_reward

    def _load_model(self):
        """
        Loads the environment model.
        """
        super()._load_model()
        mujoco_arena = BasicArena()
        mujoco_arena.set_origin([0, 0, 0])

        # using the table length to determine the offset of the robot (detail at corresponding robot.py)
        self.table_full_size = [0.8, 0.8, 0.05]
        robot_offset = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        print("robot_offset: ", robot_offset)

        # determine the offset of the robot in the environment
        self.robots[0].robot_model.set_base_xpos(robot_offset)
        self.robots[0].robot_model.set_base_ori(mat2euler(quat2mat(np.array([0.707, 0, 0.707, 0]))))

        self.model = Task(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def _setup_references(self):
        """
        Sets up references to important components.
        """
        super()._setup_references()

    def _reset_internal(self):
        """
        Resets simulation internal configurations.
        """
        super()._reset_internal()

        # # initial position of end-effector
        # self.ee_initial_pos = self._eef_xpos

        # print(self.ee_initial_pos)

        # # create trajectory
        # self.trajectory = self.get_trajectory()

        # # initialize trajectory step
        # self.initial_traj_step = np.random.default_rng().uniform(
        #     low=0, high=self.num_waypoints - 1)
        # # step at which to evaluate trajectory. Must be in interval [0, num_waypoints - 1]
        # self.traj_step = self.initial_traj_step

        # # set first trajectory point
        # self.traj_pt = self.trajectory.eval(self.traj_step)
        # self.traj_pt_vel = self.trajectory.deriv(self.traj_step)

        # # give controller access to robot (and its measurements)
        # if self.robots[0].controller.name == "HMFC":
        #     self.robots[0].controller.set_robot(self.robots[0])

        # # initialize controller's trajectory
        # self.robots[0].controller.traj_pos = self.traj_pt
        # self.robots[0].controller.traj_ori = T.quat2axisangle(self.goal_quat)

        # # get initial joint positions for robot
        # init_qpos = self._get_initial_qpos()

        # override initial robot joint positions
        # 修改：检查是否设置了相应的qpos
        # self.robots[0].set_robot_joint_positions(init_qpos)

        # update controller with new initial joints
        # self.robots[0].controller.update_initial_joints(init_qpos)

    # def get_trajectory(self):
    #     """
    #     Calculates a trajectory between two waypoints on the torso. The waypoints are extracted from a grid on the torso.
    #     The first waypoint is given at time t=0, and the second waypoint is given at t=1.
    #     If self.deterministic_trajectory is true, a deterministic trajectory along the x-axis of the torso is calculated.

    #     Args:

    #     Returns:
    #         (klampt.model.trajectory Object):  trajectory
    #     """

    #     # if self.deterministic_trajectory:
    #     # fix trajectory
    #     start_point = [0.062, -0.020,  0.896]
    #     end_point = [-0.032, -0.075,  0.896]

    #     # start_point = [grid[0, 0], grid[1, 4], self._torso_xpos[-1] + self.top_torso_offset]
    #     # end_point = [grid[0, int(self.grid_pts / 2) - 1], grid[1, 5], self._torso_xpos[-1] + self.top_torso_offset]
    #     # else:

    #     #     start_point = self._get_waypoint(grid)
    #     #     end_point = self._get_waypoint(grid)

    #     milestones = np.array([start_point, end_point])
    #     self.num_waypoints = np.size(milestones, 0)

    #     return trajectory.Trajectory(milestones=milestones)

    # def _get_initial_qpos(self):
    #     """
    #     Calculates the initial joint position for the robot based on the initial desired pose (self.traj_pt, self.goal_quat).
    #     If self.initial_probe_pos_randomization is True, Guassian noise is added to the initial position of the probe.

    #     Args:

    #     Returns:
    #         (np.array): n joint positions
    #     """
    #     # array([ 0.03595427, -0.03523952,  0.896     ])
    #     # array([ 0.02200533, -0.04340114,  0.896     ])
    #     pos = np.array(self.traj_pt)

    #     # toolbox_pos
    #     # array([-0.50200533,  0.06840114,  0.062     ])
    #     pos = self._convert_robosuite_to_toolbox_xpos(pos)
    #     ori_euler = mat2euler(quat2mat(self.goal_quat))

    #     # desired pose
    #     T = SE3(pos) * SE3.RPY(ori_euler)

    #     # find initial joint positions
    #     # trans Transform to qpos
    #     if self.robots[0].name == "UR5e":
    #         robot = rtb.models.DH.UR5()
    #         sol = robot.ikine_min(T, q0=self.robots[0].init_qpos)
    #         # flip last joint around (pi)
    #         sol.q[-1] -= np.pi
    #         print("Initial joint positions: ", sol.q)
    #         return sol.q

    #     elif self.robots[0].name == "Tendon":
    #         return [0, 0, 0, 0, 0, 0]
    #         # return sol.q

    #     elif self.robots[0].name == "Panda":
    #         robot = rtb.models.DH.Panda()
    #         sol = robot.ikine_min(T, q0=self.robots[0].init_qpos)
    #         return sol.q

    # def _convert_robosuite_to_toolbox_xpos(self, pos):
    #     """
    #     Converts origin used in robosuite to origin used for robotics toolbox. Also transforms robosuite world frame (vectors x, y, z) to
    #     to correspond to world frame used in toolbox.

    #     Args:
    #         pos (np.array): position (x,y,z) given in robosuite coordinates and frame

    #     Returns:
    #         (np.array):  position (x,y,z) given in robotics toolbox coordinates and frame
    #     """
    #     xpos_offset = self.robots[0].robot_model.base_xpos_offset["table"](
    #         self.table_full_size[0])[0]
    #     zpos_offset = self.robots[0].robot_model.top_offset[-1] - 0.016

    #     # the numeric offset values have been found empirically, where they are chosen so that
    #     # self._eef_xpos matches the desired position.
    #     if self.robots[0].name == "UR5e":
    #         return np.array([-pos[0] + xpos_offset + 0.08, -pos[1] + 0.025, pos[2] - zpos_offset + 0.15])

    #     if self.robots[0].name == "Panda":
    #         return np.array([pos[0] - xpos_offset - 0.06, pos[1], pos[2] - zpos_offset + 0.111])

    #     if self.robots[0].name == "Tendon":
    #         # 关键：需要修改
    #         return np.array([pos[0], pos[1], pos[2]])
