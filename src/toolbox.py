import numpy as np
import time

import mujoco
import mujoco.viewer

from my_environments import BasicEnv, SingleEnv
import numpy as np
import mujoco
import mujoco.viewer
import os
import sys
import time  # 导入 time 模块用于添加延迟


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import alphashape


def getliverpoint():
    # 1. 加载模型
    try:
        model_path = 'robot.xml'
        model = mujoco.MjModel.from_xml_path(model_path)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"加载模型出错: {e}")
        exit()

    # 2. 获取肝脏 geom 的 ID
    liver_geom_name = "body_liver"
    try:
        liver_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, liver_geom_name)
        if liver_geom_id == -1:
            print(f"Geom '{liver_geom_name}' 未找到.")
            exit()
    except Exception as e:
        print(f"查找 geom ID '{liver_geom_name}' 时出错: {e}")
        exit()

    # 3. 检查 geom 是否为 mesh 类型
    if model.geom_type[liver_geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
        print(f"Geom '{liver_geom_name}' 不是网格 (mesh) 类型. 其类型为 {model.geom_type[liver_geom_id]}.")
        exit()

    # 4. 获取与此 geom 关联的 mesh 的 ID
    mesh_id = model.geom_dataid[liver_geom_id]
    if mesh_id == -1:
        print(f"Geom '{liver_geom_name}' 没有关联的网格数据.")
        exit()

    # 5. 获取 mesh 的顶点信息
    vert_adr = model.mesh_vertadr[mesh_id]  # 这是起始行索引
    vert_num = model.mesh_vertnum[mesh_id]  # 这是行数

    # --- 开始调试输出 ---
    print(f"--- 调试信息 ---")
    print(f"Geom 名称: {liver_geom_name}, Geom ID: {liver_geom_id}")
    print(f"此 geom 的 Mesh ID: {mesh_id}")
    print(f"预期顶点数量 (vert_num): {vert_num}")
    print(f"顶点起始行索引 (vert_adr): {vert_adr}")

    print(f"模型中总网格数 (model.nmesh): {model.nmesh}")
    print(f"model.mesh_vert 数组的形状: {model.mesh_vert.shape}")  # 例如 (102833, 3)

    total_vertices_in_model = model.mesh_vert.shape[0]
    print(f"模型中加载的总顶点数: {total_vertices_in_model}")

    # 直接的行切片索引
    start_row_index = vert_adr
    end_row_index = vert_adr + vert_num  # 切片时不包含此行

    print(f"期望的切片起始行索引: {start_row_index}")
    print(f"期望的切片结束行索引 (不包含): {end_row_index}")

    if start_row_index >= total_vertices_in_model:
        print(f"错误: 顶点起始行索引 ({start_row_index}) 超出或等于模型总顶点数 ({total_vertices_in_model}).")
        exit()
    if end_row_index > total_vertices_in_model:  # 注意这里是 > 而不是 >=，因为end_row_index是不包含的
        print(f"警告: 期望的切片结束行索引 ({end_row_index}) 超出模型总顶点数 ({total_vertices_in_model}). 将只切片到末尾。")
        # end_row_index = total_vertices_in_model # 确保不越界 (Python切片本身会处理，但明确一下)

    # 正确的切片方式
    try:
        local_vertices = model.mesh_vert[start_row_index: end_row_index]
        print(f"提取出的 local_vertices 的形状: {local_vertices.shape}")
    except IndexError as e:
        print(f"提取 local_vertices 时发生 IndexError: {e}")
        print(f"这不应该发生，如果前面的索引检查正确的话。请检查 vert_adr ({vert_adr}) 和 vert_num ({vert_num}) 是否合理。")
        exit()
    print(f"--- 结束调试输出 ---")

    if local_vertices.shape[0] != vert_num:
        print(f"严重错误: 提取的顶点数 ({local_vertices.shape[0]}) 与期望的顶点数 ({vert_num}) 不符！")
        exit()

    if local_vertices.shape[1] != 3:
        print(f"严重错误: 提取的顶点数据不是3列 (x,y,z)，形状为: {local_vertices.shape}")
        exit()

    # local_vertices 现在应该是正确的 (vert_num, 3) 形状，无需 reshape

    # (其余坐标转换代码...)
    if local_vertices.shape[0] > 0:
        mujoco.mj_forward(model, data)
        geom_world_pos = data.geom_xpos[liver_geom_id]
        geom_world_orient_mat = data.geom_xmat[liver_geom_id].reshape((3, 3))
        world_vertices = geom_world_pos + (geom_world_orient_mat @ local_vertices.T).T

        print(f"成功提取并转换 geom '{liver_geom_name}' 的 {local_vertices.shape[0]} 个顶点.")
        # print(f"前5个世界坐标顶点:\n{world_vertices[:5]}")
    elif vert_num > 0:
        print(f"错误：期望为 geom '{liver_geom_name}' 提取 {vert_num} 个顶点，但最终 local_vertices 为空或形状不正确。")
    else:  # vert_num is 0
        print(f"信息: geom '{liver_geom_name}' 的顶点数 (vert_num) 为 0。")

    if local_vertices.shape[0] > 0:
        mujoco.mj_forward(model, data)
        geom_world_pos = data.geom_xpos[liver_geom_id]
        geom_world_orient_mat = data.geom_xmat[liver_geom_id].reshape((3, 3))
        world_vertices = geom_world_pos + (geom_world_orient_mat @ local_vertices.T).T

        print(f"成功提取并转换 geom '{liver_geom_name}' 的 {local_vertices.shape[0]} 个顶点.")

        try:
            # np.save('liver_points.npy', world_vertices)
            print(f"顶点数据已保存到 liver_points.npy")
        except Exception as e:
            print(f"保存为 .npy 文件时出错: {e}")


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
    env.IsRender = True
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


# ---------------vis-----------------

def change_h(points, h, save=False, name="liver_points_h.npy"):
    noise_magnitude = 1e-5
    points[:, 2] = h
    points[:, 2] += np.random.uniform(-noise_magnitude, noise_magnitude, size=len(points))

    np.save(f"{name}", points) if save else None
    return points


def choose_point(points):
    save_point_list = []
    for point in points:
        key = input('save this point(y/n):').lower()
        if key == 'y':
            save_point_list.append(point)
        elif key == 'q':
            break
    np.save("liver_points_1055_mask_path.npy", np.array(save_point_list))


def visual_workspace(points):
    """可视化工作空间"""
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.scatter(points[:, 0], points[:, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


def visual_workspace_3d(points):
    """可视化工作空间"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r')

    # file_name2 = "waypoint_circle.npy"
    # ee_post_list = np.load(file_name2)
    ee_post_list = get_cricle()
    ax.scatter(ee_post_list[:, 0], ee_post_list[:, 1], ee_post_list[:, 2], c='b')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(0, 0.2)
    ax.set_ylim(-0.1, 0.1)
    ax.set_zlim(1.36, 1.56)
    # ax.set_xlim(0, 0.2)
    # ax.set_ylim(-0.1, 0.1)
    # ax.set_zlim(1.37, 1.57)
    plt.show()


def visual_workspace_mujoco(points, save=True):
    save_point_list = []
    try:
        # 从 .npy 文件中加载点集
        print(f"从 {points} 加载了 {len(points)} 个点。")

        # 检查加载数据的形状是否正确 (N, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            print(f"错误：点集数据的形状应为 (N, 3)，但加载到的是 {points.shape}")
            sys.exit(1)

    except Exception as e:
        print(f"加载或处理点集文件时出错：{e}")
        sys.exit(1)

    # --- MuJoCo 可视化 ---

    # 初始化 BasicEnv 环境
    # 这会加载模型并创建 MuJoCo 数据
    try:
        env = SingleEnv()
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

        print(f"正在查看器中循环显示 {len(points)} 个点...")

        for point in points:
            data.site_xpos[waypoint_site_id] = point
            viewer.sync()
            if save:
                key = input('save this point(y/n):').lower()
                if key == 'y':
                    save_point_list.append(point)
                elif key == 'q':
                    break
            # 同步查看器以更新显示

            # 添加延迟，以便您能看到每个点
            time.sleep(display_delay)
        # print(f'保存了{len(save_point_liver)}个点')
        # np.save("liver_points_1055_mask_path.npy", np.array(save_point_list))
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


def get_cricle():

    center = np.array([0.1, 0, 1.470])

    radius = 0.1
    theta = np.linspace(0, 2 * np.pi, 200, endpoint=False)  # endpoint=False 避免首尾点重复

    # 计算相对于圆心的 x 和 y 偏移量
    r_cos_theta = radius * np.cos(theta)
    r_sin_theta = radius * np.sin(theta)

    y = center[1] + r_cos_theta
    z = center[2] + r_sin_theta
    x = np.full(200, center[0])

    circle_points = np.column_stack((x, y, z))
    return circle_points


def calculate_point_set_distances():
    from scipy.spatial import cKDTree
    import numpy as np
    set_a = np.load('waypoint_circle.npy')
    set_b = get_cricle()
    """计算两个点集之间的平均最近点距离和豪斯多夫距离。"""
    if set_a is None or set_b is None:
        return None
    tree_b = cKDTree(set_b)
    tree_a = cKDTree(set_a)
    distances_a_to_b, _ = tree_b.query(set_a, k=1)
    distances_b_to_a, _ = tree_a.query(set_b, k=1)

    print("avg_dist_A_to_B", round(np.mean(distances_a_to_b)*100, 2))
    print("avg_dist_B_to_A", round(np.mean(distances_b_to_a)*100, 2))
    print("hausdorff_A_to_B", round(np.max(distances_a_to_b)*100, 2))
    print("hausdorff_B_to_A", round(np.max(distances_b_to_a)*100, 2))
    print("hausdorff_symmetric", round(max(np.max(distances_a_to_b), np.max(distances_b_to_a))*100, 2))

    print("chamfer_distance_L1", round(np.mean(distances_a_to_b) + np.mean(distances_b_to_a), 2))


def draw():
    save_point_list = []
    try:
        # 从 .npy 文件中加载点集
        print(f"从 {points} 加载了 {len(points)} 个点。")

        # 检查加载数据的形状是否正确 (N, 3)
        if points.ndim != 2 or points.shape[1] != 3:
            print(f"错误：点集数据的形状应为 (N, 3)，但加载到的是 {points.shape}")
            sys.exit(1)

    except Exception as e:
        print(f"加载或处理点集文件时出错：{e}")
        sys.exit(1)

    # --- MuJoCo 可视化 ---

    # 初始化 BasicEnv 环境
    # 这会加载模型并创建 MuJoCo 数据
    try:
        env = SingleEnv()
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

        print(f"正在查看器中循环显示 {len(points)} 个点...")

        for point in points:
            data.site_xpos[waypoint_site_id] = point
            viewer.sync()
            # 同步查看器以更新显示

            # 添加延迟，以便您能看到每个点
            time.sleep(display_delay)
        # print(f'保存了{len(save_point_liver)}个点')
        np.save("liver_points_1055_mask_path.npy", np.array(save_point_list))
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


def calculate_distances_to_line():
    axis = 'y'
    if axis == 'y':
        line_x = 0.1
        line_z = 1.46
        npy_file_path = "line.npy"
        points = np.load(npy_file_path)

        # 提取 X 和 Z 坐标
        x_coords = points[:, 0]
        z_coords = points[:, 2]

        # 计算与直线 XY 坐标的差值
        dx = x_coords - line_x
        dz = z_coords - line_z

        # 计算距离 (使用 NumPy 的向量化操作)
        distances = np.sqrt(dx**2 + dz**2)

        print(round(np.mean(distances)*100, 2))
        print(round(np.max(distances)*100, 2))
        print(round(np.min(distances)*100, 2))
    if axis == 'z':
        line_x = 0.1
        line_y = 0
        npy_file_path = "line2.npy"
        points = np.load(npy_file_path)

        # 提取 X 和 Y 坐标
        x_coords = points[:, 0]
        y_coords = points[:, 1]

        # 计算与直线 XY 坐标的差值
        dx = x_coords - line_x
        dy = y_coords - line_y

        # 计算距离 (使用 NumPy 的向量化操作)
        distances = np.sqrt(dx**2 + dy**2)

        print(round(np.mean(distances)*100, 2))
        print(round(np.max(distances)*100, 2))
        print(round(np.min(distances)*100, 2))

    return distances


if __name__ == '__main__':
    env = SingleEnv()
    num_samples = int(2e+4)
    # file_name = "body_client_body.npy"
    # file_name = "waypoint2.npy"
    # file_name = "step2_waypoint.npy"
    # file_name = "liver_loop_path.npy"
    # file_name = "force_list.npy"

    # file_name = "ws_1.npy"
    # file_name = "waypoint_line.npy"
    file_name = "circle.npy"
    #
    # reachable_points = sample_workspace(env, num_samples)
    # np.save(f"{file_name}", reachable_points)

    points = np.load(f"{file_name}")
    print(points.shape)
    # getliverpoint(points)
    # points = change_h(points, h=1.07, save=False)
    # visual_workspace(points)
    visual_workspace_3d(points)
    # calculate_point_set_distances()
    # calculate_distances_to_line()

    # visual_workspace(points)
    # visual_workspace_mujoco(points)

    # 计算每个点与 [0,0,0.5] 的欧几里得距离
    # target_point = np.array([0, 0, 0.5])
    # distances = np.sqrt(np.sum((reachable_points - target_point) ** 2, axis=1))
    # # 找到最大距离
    # max_distance = np.max(distances)
    # print("最大距离:", max_distance)
    # exit(0)
