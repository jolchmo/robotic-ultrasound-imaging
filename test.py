import time
import numpy as np
import mujoco_py
import robosuite as suite
import os


def sort_elements(root, parent=None, element_filter=None, _elements_dict=None):
    """
    Utility method to iteratively sort all elements based on @tags. This XML ElementTree will be parsed such that
    all elements with the same key as returned by @element_filter will be grouped as a list entry in the returned
    dictionary.

    Args:
        root (ET.Element): Root of the xml element tree to start recursively searching through
        parent (ET.Element): Parent of the root node. Default is None (no parent node initially)
        element_filter (None or function): Function used to filter the incoming elements. Should take in two
            ET.Elements (current_element, parent_element) and return a string filter_key if the element
            should be added to the list of values sorted by filter_key, and return None if no value should be added.
            If no element_filter is specified, defaults to self._element_filter.
        _elements_dict (dict): Dictionary that gets passed to recursive calls. Should not be modified externally by
            top-level call.

    Returns:
        dict: Filtered key-specific lists of the corresponding elements
    """
    # Initialize dictionary and element filter if None is set
    if _elements_dict is None:
        _elements_dict = {}
    if element_filter is None:
        element_filter = _element_filter

    # Parse this element
    key = element_filter(root, parent)
    if key is not None:
        # Initialize new entry in the dict if this is the first time encountering this value, otherwise append
        if key not in _elements_dict:
            _elements_dict[key] = [root]
        else:
            _elements_dict[key].append(root)

    # Loop through all possible subtrees for this XML recurisvely
    for r in root:
        _elements_dict = sort_elements(
            root=r,
            parent=root,
            element_filter=element_filter,
            _elements_dict=_elements_dict
        )

    return _elements_dict


# 创建 Tendon 机器人模型实例
# 假设你的 Tendon 类位于 robosuite.models.robots 模块中
robot_model = suite.models.robots.Tendon()

# 获取 MJCF 模型 (返回的是 lxml.etree._Element 对象)
mjcf_model = robot_model.get_model(mode="mujoco_py")


sim = mujoco_py.MjSim(mjcf_model)

# 创建查看器
viewer = mujoco_py.MjViewer(sim)


n_joints = sim.model.nq


# 初始化关节目标位置
target_qpos = np.zeros(n_joints)


print("n_joints: ", n_joints)
print("target_qpos: ", target_qpos)


self._elements = sort_elements(root=self.root)
self._joints = [e.get("name") for e in self._elements.get(
    "joints", []) if e.get("type") != "hinge"]


# 运行仿真循环
while True:
    # 设置关节目标位置 (例如，移动第一个关节)
    target_qpos[0] = np.sin(time.time())  # 使用正弦函数让第一个关节来回移动

    # 将目标位置应用到仿真
    sim.data.qpos[:] = target_qpos

    # 执行一步仿真
    sim.step()

    # 渲染场景
    viewer.render()
