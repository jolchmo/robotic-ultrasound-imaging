import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import numpy as np
import csv

# --- 全局变量，用于存储应用程序状态 ---
waypoints = []  # 存储路径点 (x, y) 坐标的列表
fig, ax = None, None  # Matplotlib 图形和坐标轴对象
canvas = None  # 嵌入到 Tkinter 中的 Matplotlib 画布
path_line_artist = None  # Matplotlib 路径线的 Artist 对象
waypoint_markers_artist = None  # Matplotlib 路径点标记的 Artist 对象

# --- 您的二维数据占位符 ---
# !!! 至关重要: 请将下面这行替换为您的实际 (N, 2) NumPy 点数组 !!!
points_2d_background = np.load('liver_points.npy')
print(points_2d_background.shape)
# 或: points_2d_background = pd.read_csv('my_data.csv').values[:, :2]
# 如果不替换，程序将使用0-100范围的随机数据，坐标轴将与您期望的不符。
# points_2d_background = np.random.rand(100, 2) * 100 # 示例: 100个在100x100区域内的随机点

# --- Matplotlib 绘图函数 ---

# 尝试设置支持中文的字体
try:
    plt.rcParams['font.family'] = [
        'simhei'
    ]
    plt.rcParams['axes.unicode_minus'] = False
    print("已尝试设置中文字体。")
except Exception as e:
    print(f"设置中文字体时出现问题 (这通常不影响功能，但图表中的中文可能无法正确显示): {e}")
    print("请确保您的系统中安装了如 SimHei, Microsoft YaHei, WenQuanYi 等中文字体，并且Matplotlib可以找到它们。")


def setup_plot(parent_frame):
    """设置 Matplotlib 图形和坐标轴，并将其嵌入到 Tkinter 框架中。"""
    global fig, ax, canvas, path_line_artist, waypoint_markers_artist

    # figsize 定义了绘图区域的物理尺寸（英寸），其宽高比会影响 ax.axis('equal') 如何调整数据界限
    fig, ax = plt.subplots(figsize=(8, 7))  # 当前设置为宽8英寸，高7英寸

    # 1. 绘制背景二维数据
    if points_2d_background is not None:
        ax.scatter(points_2d_background[:, 0], points_2d_background[:, 1],
                   c='pink', label='肝点云', alpha=0.7, zorder=1)

        x_coords = points_2d_background[:, 0]
        y_coords = points_2d_background[:, 1]

        x_min_data, x_max_data = np.min(x_coords), np.max(x_coords)
        y_min_data, y_max_data = np.min(y_coords), np.max(y_coords)

        range_x = x_max_data - x_min_data
        range_y = y_max_data - y_min_data

        # 确定每个轴的留白（padding）
        # 对数据范围大于1e-6的使用10%的留白，否则使用0.05个单位的固定留白
        # 这个0.05是基于您数据尺度在0.1左右的考量，如果数据尺度差异很大，可能需要调整
        padding_x = range_x * 0.1 if range_x > 1e-6 else 0.05
        padding_y = range_y * 0.1 if range_y > 1e-6 else 0.05

        final_x_min = x_min_data - padding_x
        final_x_max = x_max_data + padding_x
        final_y_min = y_min_data - padding_y
        final_y_max = y_max_data + padding_y

        ax.set_xlim(final_x_min, final_x_max)
        ax.set_ylim(final_y_min, final_y_max)

        # ax.axis('equal') 会在上述建议的 xlim 和 ylim 基础上进行调整
        # 以确保数据单位在屏幕上是正方形的，它可能会扩大某个轴的范围。
        ax.axis('equal')

    # 2. 初始化路径点和路径线的 Artist 对象
    waypoint_markers_artist, = ax.plot([], [], 'ro', markersize=8, label='路径点', zorder=3)
    path_line_artist, = ax.plot([], [], 'r-', linewidth=2, label='选择的路径', zorder=2)

    ax.set_xlabel("X坐标")
    ax.set_ylabel("Y坐标")
    ax.set_title("交互选择路径点")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='upper right')

    # 3. 将 Matplotlib 图形嵌入到 Tkinter 中
    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # 4. 添加 Matplotlib 导航工具栏
    toolbar = NavigationToolbar2Tk(canvas, parent_frame)
    toolbar.update()

    # 5. 连接点击事件处理器
    fig.canvas.mpl_connect('button_press_event', on_plot_click)


def update_plot_visuals():
    """使用当前的路径点和路径更新 Matplotlib 图表。"""
    if not ax or not canvas:
        return

    wp_array = np.array(waypoints)

    if not waypoints:
        waypoint_markers_artist.set_data([], [])
        path_line_artist.set_data([], [])
    else:
        waypoint_markers_artist.set_data(wp_array[:, 0], wp_array[:, 1])
        if len(waypoints) > 1:
            path_line_artist.set_data(wp_array[:, 0], wp_array[:, 1])
        else:
            path_line_artist.set_data([], [])

    canvas.draw_idle()


def on_plot_click(event):
    """处理在 Matplotlib 图表上的鼠标点击事件。"""
    if event.inaxes == ax and event.button == 1:
        x, y = event.xdata, event.ydata
        if x is not None and y is not None:
            waypoints.append((x, y))
            print(f"路径点已添加: ({x:.2f}, {y:.2f}) - 总计: {len(waypoints)}")
            update_plot_visuals()


def save_path():
    """将选择的路径点保存到文件。"""
    if not waypoints:
        messagebox.showinfo("无路径", "没有需要保存的路径点。")
        return

    filepath = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[
            ("CSV 文件", "*.csv"),
            ("NumPy 二进制文件", "*.npy"),
            ("文本文件", "*.txt"),
            ("所有文件", "*.*")
        ],
        title="保存路径文件"
    )

    if not filepath:
        return

    try:
        if filepath.endswith('.csv'):
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['X坐标', 'Y坐标'])
                for point in waypoints:
                    writer.writerow([point[0], point[1]])
        elif filepath.endswith('.npy'):
            np.save(filepath, np.array(waypoints))
        elif filepath.endswith('.txt'):
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("# 路径点坐标 (X, Y)\n")
                for point in waypoints:
                    f.write(f"{point[0]}, {point[1]}\n")
        else:
            print(f"未知文件扩展名，将尝试按类似CSV格式保存至: {filepath}")
            with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['X坐标', 'Y坐标'])
                for point in waypoints:
                    writer.writerow([point[0], point[1]])

        messagebox.showinfo("成功", f"包含 {len(waypoints)} 个路径点的路径已保存至:\n{filepath}")
    except Exception as e:
        messagebox.showerror("保存文件错误", f"发生错误: {e}")
        print(f"保存文件错误: {e}")


def clear_path():
    """清除所有选择的路径点。"""
    if not waypoints:
        messagebox.showinfo("无路径", "路径已经为空。")
        return

    if messagebox.askyesno("确认清除", "您确定要清除所有路径点吗？"):
        waypoints.clear()
        update_plot_visuals()
        print("路径已清除。")


def undo_last_waypoint():
    """移除最后添加的路径点。"""
    if waypoints:
        removed_point = waypoints.pop()
        update_plot_visuals()
        print(f"已移除最后一个路径点: ({removed_point[0]:.2f}, {removed_point[1]:.2f})")
    else:
        messagebox.showinfo("无路径点", "没有可以撤销的路径点。")
        print("没有可撤销的路径点。")


def main():
    global root
    root = tk.Tk()
    root.title("交互式路径点选择工具")
    root.geometry("900x750")

    plot_frame = tk.Frame(root)
    plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

    setup_plot(plot_frame)

    button_frame = tk.Frame(root, pady=10)
    button_frame.pack(side=tk.BOTTOM, fill=tk.X)

    btn_save = tk.Button(button_frame, text="保存路径", command=save_path, width=15, height=2)
    btn_save.pack(side=tk.LEFT, padx=10, pady=5)

    btn_clear = tk.Button(button_frame, text="清除所有路径点", command=clear_path, width=20, height=2)
    btn_clear.pack(side=tk.LEFT, padx=10, pady=5)

    btn_undo = tk.Button(button_frame, text="撤销上一个路径点", command=undo_last_waypoint, width=20, height=2)
    btn_undo.pack(side=tk.LEFT, padx=10, pady=5)

    if canvas:  # 确保 canvas 已创建
        canvas.draw()

    root.mainloop()


if __name__ == '__main__':
    main()
