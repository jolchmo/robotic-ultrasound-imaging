import numpy as np


def rotate_quaternion(quaternion, axis, angle_degrees):
    """
    绕指定轴旋转四元数。

    Args:
      quaternion: 一个包含四个浮点数的列表或NumPy数组，代表原始四元数 (w, x, y, z)。
      axis: 一个包含三个浮点数的列表或NumPy数组，代表旋转轴的单位向量 (x, y, z)。
      angle_degrees: 旋转角度，单位为度数。

    Returns:
      一个包含四个浮点数的NumPy数组，代表旋转后的四元数 (w, x, y, z)。
    """

    # 将输入转换为NumPy数组，以方便计算
    quaternion = np.array(quaternion, dtype=np.float64)
    axis = np.array(axis, dtype=np.float64)

    # 确保轴向量是单位向量
    axis = axis / np.linalg.norm(axis)

    # 将角度转换为弧度
    angle_radians = np.radians(angle_degrees)

    # 计算旋转四元数
    half_angle = angle_radians / 2
    w = np.cos(half_angle)
    x = axis[0] * np.sin(half_angle)
    y = axis[1] * np.sin(half_angle)
    z = axis[2] * np.sin(half_angle)
    rotation_quaternion = np.array([w, x, y, z])

    # 四元数乘法
    q1 = quaternion
    q2 = rotation_quaternion

    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    rotated_quaternion = np.array([w, x, y, z])

    # 规范化结果四元数 (可选，但推荐)
    rotated_quaternion = rotated_quaternion / np.linalg.norm(rotated_quaternion)

    return [round(val, 5) for val in rotated_quaternion]


# 示例用法：
initial_quaternion = [0, 0, 0, 1]  # 单位四元数
rotation_axis = [0, 1, 0]  # Y轴
rotation_angle = -90

rotated_quaternion = rotate_quaternion(initial_quaternion, rotation_axis, rotation_angle)

print(f"原始四元数: {initial_quaternion}")
print(f"旋转轴: {rotation_axis}")
print(f"旋转角度: {rotation_angle} 度")
print(f"旋转后的四元数: {rotated_quaternion}")
