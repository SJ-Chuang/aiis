from fvcore.common.config import CfgNode as _CfgNode

__all__ = ["ParamNode", "get_params"]

class ParamNode(_CfgNode):
    def __setattr__(self, name, value):
        if self.is_frozen():
            raise AttributeError(
                "Attempted to set {} to {}, but ParamNode is immutable".format(
                    name, value
                )
            )
        self[name] = value
    def todict(self):
        def convert_to_dict(cfg_node, key_list=[]):
            if not isinstance(cfg_node, ParamNode):
                return cfg_node
            else:
                cfg_dict = dict(cfg_node)
                for k, v in cfg_dict.items():
                    cfg_dict[k] = convert_to_dict(v, key_list + [k])
                return cfg_dict
        return convert_to_dict(self)

params = ParamNode()

params.AUTH = ParamNode()
params.AUTH.URL = None
params.AUTH.USERNAME = None
params.AUTH.PASSWORD = None

params.INPUT = ParamNode()
params.INPUT.FORMAT = "BGR"

params.MODULE = ParamNode()
params.MODULE.SPACING_ON = True

params.MODULE.SPACING_HEAD = ParamNode()
params.MODULE.SPACING_HEAD.E2J_THRESH = 30
params.MODULE.SPACING_HEAD.E2E_THRESH = 50

def get_params() -> ParamNode:
    """
    Get a copy of the default parameters.

    Returns:
        a ParamNode instance.
    """

    return params.clone()