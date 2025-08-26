import os
import robosuite
from robosuite.models.objects import MujocoXMLObject

class BananaObject(MujocoXMLObject):
    def __init__(self, name="banana", **kwargs):
        abs_xml = os.path.join(
            os.path.dirname(robosuite.__file__),
            "models", "assets", "objects", "banana", "banana.xml"
        )
        super().__init__(fname=abs_xml, name=name, joints=None, **kwargs)
