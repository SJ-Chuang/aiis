from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from collections import UserDict
from . import utils
from .config import get_cfg
import os, json

__all__ = [
    'ModuleBase', 'ModuleList',
    'SpacingModule', 'HookAngleModule', 'OverlapModule'
]

MODULE_REGISTRY = utils.Registry("MODULE")

dirname = os.path.dirname(__file__)

class ModuleBase:
    def __init__(self, cfg, name="ModuleBase"):
        self.model = DefaultPredictor(cfg)
        self.name = name
        self._results = []
    
    def __call__(self, color, depth=None, coord=None, **cfg):
        """
        Sequentially apply transform an image to a dict with an actionable parameters.
        Args:
            color (np.ndarray): a color image.
            depth (np.ndarray): a depth map.
            coord (np.ndarray): 3d coordinates.
            
        Returns:
            parameters (dict): a dict of inspection parameters.
        """
        inputs = self.preprocessing(color, depth, coord, **cfg)
        predictions = self.predict(inputs, **cfg)
        parameters = self.calculate(predictions, **cfg)
        visualization = self.visualize(color, parameters, coord, **cfg)
        return visualization
    
    def predict(self, inputs, **cfg):
        """
        Args:
            inputs (np.ndarray): inputs of shape (H, W, C).

        Returns:
            predictions (dict): the output of the model for one image only.
        """
        predictions = self.model(inputs)
        return predictions
    
    def preprocessing(self, color, depth, coord, **cfg):
        """
        Pre-processing the custom input image.
        Args:
            color (np.ndarray): a color image.
            depth (np.ndarray): a depth map.
            coord (np.ndarray): 3d coordinates.
            
        Returns:
            A pre-processed inputs.
        """
        format = cfg['INPUT']['FORMAT'] if cfg else 'BGR'
        assert format in ['BGR', 'RGB'], f"Input format must be 'BGR' or 'RGB', not {format}"
        inputs = color[:,:,::-1] if format == "RGB" else color
        return inputs
    
    def calculate(self, predictions, **cfg):
        """
        Args:
            predictions (dict): the output of the model.
            
        Returns:
            parameters for inspection.
        """
        parameters = predictions
        return parameters
    
    def visualize(self, background, parameters, coord=None, **cfg):
        """
        Args:
            background (np.ndarray): the background image.
            parameters (dict): a dict of inspection parameters.
            coord (np.ndarray): 3d coordinates.
            
        Returns:
            vis (np.ndarray): parameter visualization.
        """
        vis = background
        return vis

class ModuleList(ModuleBase):
    def __init__(self, modules, cfg=None):
        """
        Args:
            modules (List[ModuleBase or str]): a list of defined ModuleBase, i.e., [SpacingModule(cfg), "HookAngleModule"].
        """
        self._modules = []
        for module in modules:
            if isinstance(module, ModuleBase):
                self._modules.append(module)
            
            elif isinstance(module, str):
                self._modules.append(MODULE_REGISTRY.get(module)())
            
            else:
                assert False, "modules must be an object of ModuleBase or a name of registered module."
        
        self.name = "ModuleList"
        
    def __call__(self, color, depth=None, coord=None, **cfg):
        parameters = []
        for module in self._modules:
            inputs = module.preprocessing(color, depth, coord, **cfg)
            predictions = module.predict(inputs, **cfg)
            parameters.append(module.calculate(predictions, **cfg))
        
        visualization = color
        for module, params in zip(self._modules, parameters):
            visualization = module.visualize(visualization, params, coord, **cfg)
        
        return visualization

@MODULE_REGISTRY.register()
class SpacingModule(ModuleBase):
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = get_cfg()
        if cfg.MODEL.WEIGHTS is None:
            cfg.MODEL.WEIGHTS = os.path.join(dirname, './weights/spacing.pth')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        super().__init__(cfg, "SpacingModule")
    
    def calculate(self, predictions, delta_e2j=30, delta_e2e=50, **cfg):
        if cfg:
            delta_e2j = cfg['MODULE']['SPACING_HEAD']['E2J_THRESH']
            delta_e2e = cfg['MODULE']['SPACING_HEAD']['E2E_THRESH']
        pred_masks = predictions['instances'].pred_masks
        pred_classes = predictions['instances'].pred_classes
        junctions = utils.find_centroid_by_mask(pred_masks[pred_classes==0])
        enpoints, line_masks = utils.find_endpoint_by_mask(pred_masks[pred_classes==1].cpu())
        links = utils.find_nearest_link(
            junctions, enpoints,
            line_masks=line_masks,
            max_e2j_dist=delta_e2j, max_e2e_dist=delta_e2e,
            return_index=False).numpy().astype(int)
        return links
    
    def visualize(self, background, links, coord=None, **cfg):
        return utils.vis_link(background, links, coord=coord, **cfg)

@MODULE_REGISTRY.register()
class HookAngleModule(ModuleBase):
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'))
            cfg.MODEL.WEIGHTS = os.path.join(dirname, './weights/hook_angle.pth')
            
        if cfg.MODEL.WEIGHTS is None:
            cfg.MODEL.WEIGHTS = os.path.join(dirname, './weights/hook_angle.pth')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4
        
        super().__init__(cfg, "HookAngleModule")
    
    def visualize(self, background, predictions, coord=None, **cfg):
        metadata = UserDict({'thing_classes': ["Unknown Angle", f"90{chr(176)}", f"135{chr(176)}", f"180{chr(176)}"]})
        return utils.vis_prediction(background, predictions, metadata)

@MODULE_REGISTRY.register()
class OverlapModule(ModuleBase):
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file('COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'))
            cfg.MODEL.WEIGHTS = os.path.join(dirname, './weights/overlap.pth')
            
        if cfg.MODEL.WEIGHTS is None:
            cfg.MODEL.WEIGHTS = os.path.join(dirname, './weights/overlap.pth')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        super().__init__(cfg, "OverlapModule")
    
    def visualize(self, background, predictions, coord=None, **cfg):
        metadata = UserDict({'thing_classes': ["Edge", "Overlap"]})
        return utils.vis_prediction(background, predictions, metadata)