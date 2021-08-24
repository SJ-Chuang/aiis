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

```python
from tools.client.config import get_cfg
from tools.client.utils import Poster
import cv2

# Set configurations
cfg = get_cfg()
cfg.AUTH.URL = "http://www.aiis.net"
cfg.AUTH.USERNAME = "username"
cfg.AUTH.PASSWORD = "password"

# Build a Poster
poster = Poster(cfg)

# Read an image
image = cv2.imread('image.png')

# Send a post request to url
poster.post(image)
```

