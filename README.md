# AIiS (Autonomous Inspection in Steelscapes)

A Django project for back-end automatic rebar inspection service.

## Installation

*Torch*

```shell
pip3 install torch==1.8.2+cu111 torchvision==0.9.2+cu111 torchaudio==0.8.2 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

*Detectron2*

```shell
python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html
```

## Client Tools

Download the source code from git and move change the working directory.

```shell
git clone https://github.com/SJ-Chuang/aiis.git
```

```shell
cd aiis
```

The following is an example of sending a POST request to the specified URL.

```python
from aicore.client.params import get_params
from aicore.client.utils import Poster
import numpy as np
import cv2

# Parameters settings
params = get_params()
params.AUTH.URL = "http://www/aiis.net/"
params.AUTH.USERNAME = "username"
params.AUTH.PASSWORD = "password"

# Poster establishment
poster = Poster(params)

# Color image reading
color = cv2.imread('aicore/modules/demo/color.png')
depth = cv2.imread('aicore/modules/demo/color.png')
coord = np.load('aicore/modules/demo/coord.npy')

# Send a POST request to params.AUTH.URL
vis = poster.post(color=color, depth=depth, coord=coord)

if vis is not None:
	vis.save("vis.png")
```

