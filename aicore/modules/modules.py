from detectron2.engine import DefaultPredictor
from . import utils
from .config import get_cfg
import os, json

__all__ = [
    'ModuleBase', 'ModuleList',
    'SpacingModule'
]

dirname = os.path.dirname(__file__)

class ModuleBase:
    def __init__(self, cfg, name="ModuleBase"):
        self.model = DefaultPredictor(cfg)
        self.name = name
        self._results = []
    
    def __call__(self, image, *args, **kwargs):
        """
        Sequentially apply transform an image to a dict with an actionable parameters.
        Args:
            image (np.ndarray): an image.
        Returns:
            parameters (dict): a dict of inspection parameters.
        """
        image = self.preprocessing(image, *args, **kwargs)
        predictions = self.predict(image, *args, **kwargs)
        parameters = self.calculate(predictions, *args, **kwargs)
        visualization = self.visualize(image, parameters, *args, **kwargs)
        return visualization
    
    def predict(self, image, *args, **kwargs):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict): the output of the model for one image only.
        """
        predictions = self.model(image)
        return predictions
    
    def preprocessing(self, image, format='BGR', *args, **kwargs):
        """
        Preprocessing the custom input image.
        Args:
            image (np.ndarray): an image.
        Returns:
            A pre-processed image.
        """
        assert format in ['BGR', 'RGB'], f"Input format must be 'BGR' or 'RGB', not {format}"
        image = image[:,:,::-1] if format == "RGB" else image
        return image
    
    def calculate(self, predictions, *args, **kwargs):
        """
        Args:
            predictions (dict): the output of the model.
        Returns:
            parameters (dict): a dict of inspection parameters.
        """
        parameters = {"params": None}
        return parameters
    
    def visualize(self, background, parameters, *args, **kwargs):
        """
        Args:
            background (np.ndarray): the background image.
            parameters (dict): a dict of inspection parameters.
        Returns:
            vis (np.ndarray): parameter visualization.
        """
        vis = background
        return vis

class ModuleList(ModuleBase):
    def __init__(self, modules):
        """
        Args:
            modules (List[ModuleBase]): a list of defined ModuleBase, i.e., {'spacing': SpacingModule(cfg)}.
        """
        super().__init__(cfg)
    
    def __call__(self, image):
        for name, module in self._modules.items():
            self._results[name].append(module.inference_single_image(image))

class SpacingModule(ModuleBase):
    def __init__(self, cfg=None):
        if cfg is None:
            cfg = get_cfg()
        if cfg.MODEL.WEIGHTS is None:
            cfg.MODEL.WEIGHTS = os.path.join(dirname, './weights/spacing.pth')
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        super().__init__(cfg, "SpacingModule")
    
    def calculate(self, predictions, delta_e2j=30, delta_e2e=50, *args, **kwargs):
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
    
    def visualize(self, background, links, coord=None, *args, **kwargs):
        return utils.vis_link(background, links, coord=coord)


        
        
        
        
        
        