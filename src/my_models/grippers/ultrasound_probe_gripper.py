"""
Ultrasound Probe Gripper.
"""
import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel


class UltrasoundProbeGripper(GripperModel):
    """
    Ultrasound Probe Gripper.
    Args:
        idn (int or str): Number or some other unique identification string for this gripper instance
    """

    def __init__(self, idn=0):
        super().__init__("my_models/assets/grippers/ultrasound_probe_gripper.xml", idn=idn)

    def format_action(self, action):
        return action

    @property
    def init_qpos(self):
        return None

    @property
    def _important_geoms(self):
        return {
            "probe": ["probe_collision"]
        }
