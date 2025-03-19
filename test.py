# import mujoco_py
# import os
# mj_path, _ = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
# model = mujoco_py.load_model_from_path(xml_path)
# sim = mujoco_py.MjSim(model)
# print(sim.data.qpos)
# sim.step()
# print(sim.data.qpos)
import os
os.add_dll_directory("C://Users//john//.mujoco//mujoco200//bin")
import mujoco_py
import os

# Replace "your_model.xml" with the actual path to your XML file
xml_path = "./src/my_models/assets/all/allinone.xml"

# Check if the file exists
if not os.path.exists(xml_path):
    print(f"Error: XML file not found at {xml_path}")
    exit()

# Load the model
try:
    model = mujoco_py.load_model_from_path(xml_path)
    sim = mujoco_py.MjSim(model)
    viewer = mujoco_py.MjViewer(sim)
    viewer.vopt.flags[mujoco_py.enums.mjtVisFlag.mjVIS_MENU] = False  # Hide the menu

    # Run the simulation loop
    while True:
        viewer.render()  # Render the scene
        sim.step()       # Advance the simulation one step

except Exception as e:
    print(f"Error: {e}")
