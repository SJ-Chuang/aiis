from aicore.modules.modules import ModuleList
import numpy as np
from tqdm import tqdm
import os, cv2

base_dir = "/home/aiis/aiis/20210903/"

os.makedirs(os.path.join(base_dir, 'vis'), exist_ok=True)

module = ModuleList(["SpacingModule", "HookAngleModule"])

for file in tqdm(os.listdir(os.path.join(base_dir, 'color'))):
    id = os.path.splitext(file)[0]
    coord = np.load(os.path.join(*[base_dir, 'coord', id+'.npy']))
    color = cv2.imread(os.path.join(*[base_dir, 'color', id+'.png']))
    vis = module(color, coord=coord)
    cv2.imwrite(os.path.join(*[base_dir, 'vis', file]), vis)
    