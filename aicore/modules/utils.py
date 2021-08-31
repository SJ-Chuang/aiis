from detectron2.utils.visualizer import Visualizer, ColorMode
from fvcore.common.registry import Registry
from scipy.spatial.distance import cdist
from imantics import Mask
import numpy as np
import cv2, torch

__all__ = [
    'Registry',
    'read_image',
    'vis_prediction', 'vis_link',
    'find_centroid_by_mask', 'find_endpoint_by_mask', 'find_nearest_link'
]

def read_image(image_path, format='BGR'):
    """
    Read an image from its path.
    Args:
        image_path (str): a path of an image, i.e., './image.png'
        format (str): image format
    Returns:
        a numpy array
    """
    return detection_utils.read_image(image_path, format=format)

def vis_prediction(background, prediction, metadata=None):
    """
    Visualize prediction with background.
    Args:
        background (numpy.ndarray): a background image.
        prediction (dict): a dict with detectron2 prediction.
        metadata (UserDict): dataset metadata (e.g. )
        
    Returns:
        a numpy array
    """
    visualizer = Visualizer(background, metadata, scale=1, instance_mode=ColorMode.IMAGE)
    out = visualizer.draw_instance_predictions(prediction["instances"].to("cpu"))
    return out.get_image()

def vis_link(background, links, junc_color=(0, 0, 255), link_color=(0, 127, 0), coord=None, **kwarg):
    """
    Visualize links with background.
    Args:
        background (numpy.ndarray): a background image
        links (numpy.ndarray): an array of two endpoints, i.e., numpy.ndarray([[[0, 0], [10, 10]], [[0, 10], [50, 10]], ...])
        junc_color (tuple): color of junctions.
        link_color (tuple): color of links.
        coord (numpy.ndarray): an 3d coordinates of shape (H, W, 3), where H and W is the height and width of the mask, respectively
    Returns:
        a numpy array
    """
    background = background.astype(np.uint8)
    for link in np.array(links).astype(int):
        (x1, y1), (x2, y2) = link
        cv2.line(background, (x1, y1), (x2, y2), (0, 0, 0), 7)
        cv2.line(background, (x1, y1), (x2, y2), link_color, 3)
        cv2.circle(background, (x1, y1), 5, junc_color, 2)
        cv2.circle(background, (x2, y2), 5, junc_color, 2)
        if coord is not None:
            coord1 = coord[y1, x1]
            coord2 = coord[y2, x2]
            if np.sum(coord1 != 0) and np.sum(coord2 != 0):
                length = np.linalg.norm(coord1-coord2) * 100
                text = format(length, '.2f')
                ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 3)[0]
                cv2.putText(background, text, ((x1+x2)//2-ts[0]//2, (y1+y2)//2+ts[1]//2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 3)
                ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)[0]
                cv2.putText(background, text, ((x1+x2)//2-ts[0]//2, (y1+y2)//2+ts[1]//2), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,255), 1)
    return background

def find_centroid_by_mask(masks):
    """
    Find centroids from predicted masks.
    Args:
        masks (torch.tensor): a tensor of predicted masks of shape (N, H, W), where N is the number
            of mask. H and W is the height and width of the mask, respectively
    Returns:
        a torch tensor
    """
    centroids = []
    for mask in np.asarray(masks.cpu()):
        polys = Mask(mask).polygons()
        pts = [pt for contour in polys.points for pt in contour]
        if len(pts) >= 3:
            M = cv2.moments(np.array(pts))
            centroids.append([M['m10'] / M["m00"], M['m01'] / M["m00"]])
    return torch.tensor(centroids)

def find_endpoint_by_mask(masks, return_linemasks=True):
    """
    Find the two farest points as the endpoints of the each mask.
    Args:
        masks (torch.tensor): a tensor of predicted masks of shape (N, H, W), where N is the number
            of mask. H and W is the height and width of the mask, respectively
        return_linemasks (bool): if return the masks with endpoints.
    Returns:
        torch tensors
    """
    endpoints = []
    linemask_indices = []
    for m, mask in enumerate(np.asarray(masks.cpu())):
        polys = Mask(mask).polygons()
        pts = [pt for contour in polys.points for pt in contour]
        if len(pts) >= 2:
            dist_matrix = cdist(pts, pts, metric='euclidean')
            i, j = np.where(dist_matrix==dist_matrix.max())[0][:2]
            endpoints.append([pts[i], pts[j]])
            linemask_indices.append(m)
    if return_linemasks:
        return torch.tensor(endpoints), masks[linemask_indices]
    return torch.tensor(endpoints)

def find_nearest_link(juncs, lines, line_masks=None,
        max_e2j_dist=30, max_e2e_dist=50, path_thred=0.5, e2e_on=True, return_index=True):
    """
    Find the links between junctions and lines.
    Args:
        juncs (torch.tensor): a tensor of junctions of shape (N, 2), where N is the number
            of junction. Each junction is represented by a point (X, Y).
        lines (torch.tensor): a tensor of lines of shape (N, 2, 2), where N is the number
            of line. Each line is represented by two points.
        line_masks (Optional[torch.tensor]): a tensor of predicted masks of shape (N, H, W), where N is the number
            of mask. H and W is the height and width of the mask, respectively
        max_e2j_dist (int): the maximun tolerance distance between endpoints and junctions.
        max_e2e_dist (int): the maximun tolerance distance between endpoints and enpoints.
        path_thred (Optional[float]): a float between [0, 1] that filters out links with path confindence under path_thred.
        return_index (bool): if return the indices of connected junction.
    Returns:
        a torch tensor
    """
    def line_line_intersection(line1, line2):
        v1 = line1[1]-line1[0]
        v2 = line2[1]-line2[0]
        m1 = v1[1] / v1[0]
        m2 = v2[1] / v2[0]
        if m1 != m2:
            x = (line2[0, 1]-line1[0, 1] + m1 * line1[0, 0] - m2 * line2[0, 0]) / (m1 - m2)
            y = m1 * (x - line1[0, 0]) + line1[0, 1]
            intersection = torch.tensor([x, y])
            if torch.linalg.norm(intersection-line1[1]) < max_e2j_dist and torch.linalg.norm(intersection-line2[1]) < max_e2j_dist:
                return intersection
        
    links = []
    for l, line in enumerate(lines):
        # E2J link prediction
        e2j_dist_matrix = cdist(line, juncs, metric='euclidean')
        i, j = e2j_dist_matrix.argsort(1)[:,0]
        if i != j and e2j_dist_matrix[0, i] < max_e2j_dist and e2j_dist_matrix[1, j] < max_e2j_dist:
            if line_masks is not None:
                length = torch.linalg.norm(juncs[i]-juncs[j]).int()
                x = torch.linspace(juncs[i][0], juncs[j][0], length).long()
                y = torch.linspace(juncs[i][1], juncs[j][1], length).long()
                if line_masks[l, y, x].sum() / length < path_thred:
                    continue
                    
            if return_index:
                links.append([i, j])
                
            else:
                links.append(juncs[[i, j]].numpy().tolist())
        
        # E2E link prediction
        elif e2e_on:
            dist_ei, dist_ej = cdist(line, lines.view(-1, 2), metric='euclidean')
            if e2j_dist_matrix[0, i] < max_e2j_dist:
                for ej in np.where((0 < dist_ej) & (dist_ej < max_e2e_dist))[0]:
                    intersection = line_line_intersection(line, lines[ej // 2, [1 - ej % 2, ej % 2]])
                    if intersection is not None:
                        juncs = torch.cat([juncs, intersection.unsqueeze(0)], 0)
                        if return_index:
                            links.append([i, len(juncs)-1])
                            
                        else:
                            links.append(juncs[[i, -1]].numpy().tolist())
                        break
                    
            elif e2j_dist_matrix[1, j] < max_e2j_dist:
                for ei in np.where((0 < dist_ei) & (dist_ei < max_e2e_dist))[0]:
                    intersection = line_line_intersection(line[[1, 0]], lines[ei // 2, [1 - ei % 2, ei % 2]])
                    if intersection is not None:
                        juncs = torch.cat([juncs, intersection.unsqueeze(0)], 0)
                        if return_index:
                            links.append([j, len(juncs)-1])
                            
                        else:
                            links.append(juncs[[j, -1]].numpy().tolist())
                        break
            else:
                link = []
                for ej in np.where((0 < dist_ej) & (dist_ej < max_e2e_dist))[0]:
                    intersection = line_line_intersection(line, lines[ej // 2, [1 - ej % 2, ej % 2]])
                    if intersection is not None:
                        juncs = torch.cat([juncs, intersection.unsqueeze(0)], 0)
                        link.append(len(juncs)-1)
                        break
                for ei in np.where((0 < dist_ei) & (dist_ei < max_e2e_dist))[0]:
                    intersection = line_line_intersection(line[[1, 0]], lines[ei // 2, [1 - ei % 2, ei % 2]])
                    if intersection is not None:
                        juncs = torch.cat([juncs, intersection.unsqueeze(0)], 0)
                        link.append(len(juncs)-1)
                        break
                if len(link) == 2:
                    if return_index:
                        links.append(link)
                    else:
                        links.append(juncs[link].numpy().tolist())
    if return_index:
        return juncs, torch.tensor(links)
        
    return torch.tensor(links)