from django.db import models
from django.conf import settings
from django.utils import timezone
from django.contrib.auth.models import User
from .client.params import ParamNode
from .modules import ModuleList
from .modules import utils
from datetime import datetime
from PIL import Image
import numpy as np
import os, io, cv2, json, base64

class Post(models.Model):
    title = models.CharField(max_length=100, default="Post")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )
    created_date = models.DateTimeField(default=timezone.now)
    module = ModuleList(["SpacingModule", "HookAngleModule"])
    base_dir = "/home/aiis"
    
    def encode(self, img):
        img = Image.fromarray(img)
        f = io.BytesIO()
        img.save(f, format="PNG")
        content = f.getvalue()
        return base64.encodebytes(content)
    
    def decode(self, arg):
        if arg is None:
            return
            
        try:
            img = base64.b64decode(arg)
            f = io.BytesIO()
            f.write(img)
            return np.array(Image.open(f))
            
        except:
            return np.load(io.BytesIO(base64.b64decode(arg)))['data']
    def save_frameset(self, color=None, depth=None, coord=None, date=None, time=None):
        Date = datetime.strftime(datetime.now(), '%Y%m%d') if date is None else date
        Time = datetime.strftime(datetime.now(), '%H%M%S') if time is None else time
        
        save_dir = os.path.join(*[self.base_dir, self.user.username, Date])
        os.umask(0)
        os.makedirs(save_dir, mode=0o770, exist_ok=True)
        
        if isinstance(color, np.ndarray):
            os.makedirs(os.path.join(save_dir, 'color'), exist_ok=True)
            cv2.imwrite(os.path.join(*[save_dir, 'color', f'{Date}_{Time}.png']), color)
        
        if isinstance(depth, np.ndarray):
            os.makedirs(os.path.join(save_dir, 'depth'), exist_ok=True)
            cv2.imwrite(os.path.join(*[save_dir, 'depth', f'{Date}_{Time}.png']), depth)
        
        if isinstance(coord, np.ndarray):
            os.makedirs(os.path.join(save_dir, 'coord'), exist_ok=True)
            np.save(os.path.join(*[save_dir, 'coord', f'{Date}_{Time}']), coord)
            
    def publish(self, request):
        color = self.decode(request.POST.get('color'))
        depth = self.decode(request.POST.get('depth'))
        coord = self.decode(request.POST.get('coord'))
        date = request.POST.get('date')
        time = request.POST.get('time')
        self.save_frameset(color, depth, coord, date, time)
        params = ParamNode(json.loads(request.POST.get("params")))
        vis = self.encode(self.module(color=color, depth=depth, coord=coord, params=params))
        return [vis]
        
    def __str__(self):
        return f"{self.title}: {self.user}"