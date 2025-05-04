import numpy as np
import time

import mujoco
import mujoco.viewer

from my_environments import BasicEnv
import numpy as np
import mujoco
import mujoco.viewer
import os
import sys
import time  # 导入 time 模块用于添加延迟


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


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
    env.IsWaypointRandom = False
    # 预分配数组以提高效率
    reachable_points = np.zeros((num_samples, 3))
    num_joints = env.n_joints
    for i in range(num_samples):
        # 随机生成所有关节的角度，范围在关节限制内
        random_joints = np.random.uniform(0, 20, size=(num_joints))
        env.data.ctrl[:] = random_joints  # 设置所有关节的控制输入
        start_time = time.time()
        while time.time() - start_time < max_time_per_sample:
            mujoco.mj_step(env.mj_model, env.data)
            # env.render()
            # 如果系统稳定，可以提前退出（可选：添加稳定性判断条件）
            if np.all(np.abs(env.data.qvel[:6]) < 1e-3):
                break
        # env.set_waypoint(env.ee_pos)  # 设置路点位置
        reachable_points[i] = env.ee_pos.copy()  # 假设 ee_pos 是末端执行器位置
        # print(env.ee_pos)
        # input("wait")

    return reachable_points


def visual_workspace(workspace_points):
    """可视化工作空间"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


def visual_workspace_mujoco():

    # .npy 文件路径
    # 根据您之前的文件和讨论，假设要可视化 waypoint_s.npy 或 waypoint.npy
    # 请根据您的实际文件名修改这里
    waypoint_file = "waypoint_top.npy"  # 或者 "waypoint.npy"

    # 检查文件是否存在
    if not os.path.exists(waypoint_file):
        print(f"错误：点集文件 '{waypoint_file}' 未找到。")
        print("请确保 .npy 文件已上传且路径正确。")
        sys.exit(1)

    try:
        # 从 .npy 文件中加载点集
        waypoints = np.load(waypoint_file)
        print(f"从 {waypoint_file} 加载了 {len(waypoints)} 个点。")

        # 检查加载数据的形状是否正确 (N, 3)
        if waypoints.ndim != 2 or waypoints.shape[1] != 3:
            print(f"错误：点集数据的形状应为 (N, 3)，但加载到的是 {waypoints.shape}")
            sys.exit(1)

    except Exception as e:
        print(f"加载或处理点集文件时出错：{e}")
        sys.exit(1)

    # --- MuJoCo 可视化 ---

    # 初始化 BasicEnv 环境
    # 这会加载模型并创建 MuJoCo 数据
    try:
        env = BasicEnv()
        model = env.mj_model  # 获取 MuJoCo 模型对象
        data = env.data     # 获取 MuJoCo 数据对象
        # 获取用于可视化的 waypoint site 的 ID
        # BasicEnv 应该已经获取了这个 ID 并存储在 self.waypoint_geom_id 中
        waypoint_site_id = env.waypoint_geom_id

        if waypoint_site_id == -1:
            print("错误：在模型中未找到名为 'robot0_waypoint' 的 site。无法可视化点集。")
            print("请检查 robot.xml 文件是否包含 <site name='robot0_waypoint' .../>")
            sys.exit(1)

        # 启动 MuJoCo 查看器
        viewer = mujoco.viewer.launch_passive(model, data)
        print("MuJoCo 查看器已启动。按 ESC 键退出。")

        # 循环遍历点集并在查看器中显示
        # 您可以调整这里的逻辑，例如一次显示多个点（如果 XML 中有多个可视化 site）
        # 或者像这里一样，逐点更新显示。
        display_delay = 0.1  # 每个点在屏幕上显示的时间（秒）

        print(f"正在查看器中循环显示 {len(waypoints)} 个点...")

        for point in waypoints:
            # 更新 waypoint site 的位置
            # site 的位置存储在 mjData 中
            data.site_xpos[waypoint_site_id] = point

            # 同步查看器以更新显示
            viewer.sync()

            # 添加延迟，以便您能看到每个点
            time.sleep(display_delay)

            # 检查查看器是否还在运行
            if not viewer.is_running():
                print("查看器已关闭。")
                break

        print("点集显示完毕。查看器将保持打开状态直到您手动关闭。")

        # 保持查看器打开，直到用户关闭它
        while viewer.is_running():
            viewer.sync()
            time.sleep(model.opt.timestep)  # 可以使用模型的 timestep 作为同步间隔

        # 关闭查看器
        viewer.close()

    except Exception as e:
        print(f"MuJoCo 可视化过程中发生错误：{e}")
        # 确保在出错时关闭查看器（如果已打开）
        if 'viewer' in locals() and viewer.is_running():
            viewer.close()


if __name__ == '__main__':
    env = BasicEnv()
    num_samples = int(2e+4)
    # num_samples = int(1e+3)

    # reachable_points = sample_workspace(env, num_samples)
    # np.save("waypoint_top.npy", reachable_points)

    # reachable_points = np.load("waypoint_top.npy")
    # visual_workspace(reachable_points)

    visual_workspace_mujoco()
    # 计算每个点与 [0,0,0.5] 的欧几里得距离
    # target_point = np.array([0, 0, 0.5])
    # distances = np.sqrt(np.sum((reachable_points - target_point) ** 2, axis=1))
    # # 找到最大距离
    # max_distance = np.max(distances)
    # print("最大距离:", max_distance)
    # exit(0)
