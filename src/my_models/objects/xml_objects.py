import numpy as np

from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import array_to_string


class ClientBodyObject(MujocoXMLObject):
    """
    Client body object
    """

    def __init__(self, name):
        super().__init__("my_models/assets/objects/client_body.xml", name=name)


class SoftTorsoObject(MujocoXMLObject):
    """
    Soft torso object
    """

    def __init__(self, name, damping=None, stiffness=None):
        super().__init__("my_models/assets/objects/soft_human_torso.xml", name=name, duplicate_collision_geoms=False)

        self.damping = damping
        self.stiffness = stiffness

        if self.damping is not None:
            self.set_damping(damping)
        if self.stiffness is not None:
            self.set_stiffness(stiffness)

    def _get_composite_element(self):
        return self._obj.find("./composite")

    def set_damping(self, damping):
        """
        Helper function to override the soft body's damping directly in the XML
        Args:
            damping (float, must be greater than zero): damping parameter to override the ones specified in the XML
        """
        assert damping > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        stiffness = float(solref_str[0])

        solref = np.array([stiffness, -damping])
        composite.set('solrefsmooth', array_to_string(solref))

    def set_stiffness(self, stiffness):
        """
        Helper function to override the soft body's stiffness directly in the XML
        Args:
            stiffness (float, must be greater than zero): stiffness parameter to override the ones specified in the XML
        """
        assert stiffness > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        damping = float(solref_str[1])

        solref = np.array([-stiffness, damping])
        composite.set('solrefsmooth', array_to_string(solref))


class SoftBoxObject(MujocoXMLObject):
    """
    Soft box object
    """

    def __init__(self, name, damping=None, stiffness=None):
        super().__init__("my_models/assets/objects/soft_box.xml", name=name, duplicate_collision_geoms=False)

        self.damping = damping
        self.stiffness = stiffness

        if self.damping is not None:
            self.set_damping(damping)
        if self.stiffness is not None:
            self.set_stiffness(stiffness)

    def _get_composite_element(self):
        return self._obj.find("./composite")

    def set_damping(self, damping):
        """
        Helper function to override the soft body's damping directly in the XML
        Args:
            damping (float, must be greater than zero): damping parameter to override the ones specified in the XML
        """
        assert damping > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        stiffness = float(solref_str[0])

        solref = np.array([stiffness, -damping])
        composite.set('solrefsmooth', array_to_string(solref))

    def set_stiffness(self, stiffness):
        """
        Helper function to override the soft body's stiffness directly in the XML
        Args:
            stiffness (float, must be greater than zero): stiffness parameter to override the ones specified in the XML
        """
        assert stiffness > 0, 'Damping must be greater than zero'

        composite = self._get_composite_element()
        solref_str = composite.get('solrefsmooth').split(' ')
        damping = float(solref_str[1])

        solref = np.array([-stiffness, damping])
        composite.set('solrefsmooth', array_to_string(solref))


class BoxObject(MujocoXMLObject):
    """
    Box object
    """

    def __init__(self, name):
        super().__init__("my_models/assets/objects/box.xml", name=name)
