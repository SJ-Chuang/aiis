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
from aicore.client.params import get_params, ParamNode
from aicore.client.utils import Poster
import cv2

# Parameters settings
params = get_params()
params.AUTH.URL = "http://140.112.13.4/aicore/upload/"
params.AUTH.USERNAME = "aiis"
params.AUTH.PASSWORD = "aiis"

# Poster establishment
poster = Poster(params)

# Color image reading
color = cv2.imread('aicore/modules/demo/color.png')

# Send a POST request to params.AUTH.URL
poster.post(color)
```

