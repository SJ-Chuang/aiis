from aicore.modules import get_cfg
from aicore.modules import SpacingModule
import aicore.modules.utils as utils
import numpy as np
import cv2, time

cfg = get_cfg()
t = time.time()
module = SpacingModule(cfg)
print(f"Loading took {time.time() - t} sec.")
color = cv2.imread('aicore/modules/demo/color.png')

times = 1000

predict_costs = []
for _ in range(times):
    t = time.time()
    predictions = module.predict(color)
    predict_costs.append(time.time() - t)
    # print(f"Predcition took {time.time() - t} sec.")

    t = time.time()
    parameters = module.calculate(predictions)
    # print(f"Calculation took {time.time() - t} sec.")

print(f"The average time it took to run a {times} predictions is {np.mean(predict_costs):.3f} sec.")

print('done')