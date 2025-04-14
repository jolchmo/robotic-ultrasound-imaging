
import numpy as np
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import array_to_string, string_to_array


class BasicArena(Arena):
    """
    Workspace that contains an empty table.
    Args:
        table_full_size (3-tuple): (L,W,H) full dimensions of the table
        table_friction (3-tuple): (sliding, torsional, rolling) friction parameters of the table
        table_offset (3-tuple): (x,y,z) offset from center of arena when placing table.
            Note that the z value sets the upper limit of the table
        has_legs (bool): whether the table has legs or not
        xml (str): xml file to load arena
    """

    def __init__(
        self, xml="my_models/assets/arenas/basic.xml",
    ):
        super().__init__(xml)
