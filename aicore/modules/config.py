from detectron2.config import CfgNode
from detectron2 import model_zoo
import os

__all__ = ['get_cfg', 'save_cfg']

def get_cfg() -> CfgNode:
    """
    Get a copy of the default config.
    Returns:
        a detectron2 CfgNode instance.
    """
    from detectron2.config.defaults import _C
    cfg = _C.clone()
    cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'))
    cfg.MODEL.WEIGHTS = None
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.DEVICE = 'cuda'
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    
    return cfg

def save_cfg(cfg, save_name):
    """
    Save config to a .yaml file.
    Args:
        cfg (detectron2.config.CfgNode): a detectron2 CfgNode instance.
        save_name (str): path to save config, i.e., 'config.yaml'.
    """
    assert os.path.splitext(save_name)[-1] == '.yaml', "The extension of the saved name must be .yaml"
    os.makedirs(os.path.dirname(save_name), exist_ok=True)
    with open(save_name, 'w') as f:
        f.write(cfg.dump())
    f.close()

def check_cfg(cfg):
    cfg.merge_from_file(model_zoo.get_config_file(model_cfg))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_cfg)
    return cfg