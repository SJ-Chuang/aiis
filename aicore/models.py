from django.db import models
from django.conf import settings
from django.utils import timezone
from django.contrib.auth.models import User
from .client.params import ParamNode
from .modules import SpacingModule
from .modules import utils
from PIL import Image
import numpy as np
import io, json, base64

class Post(models.Model):
    title = models.CharField(max_length=100, default="Post")
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE
    )
    created_date = models.DateTimeField(default=timezone.now)
    module = SpacingModule()
    
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
            return np.load(io.BytesIO(base64.b64decode(arg)))

    def publish(self, request):
        color = self.decode(request.POST.get('color'))
        depth = self.decode(request.POST.get('depth'))
        coord = self.decode(request.POST.get('coord'))
        params = ParamNode(json.loads(request.POST.get("params")))
        vis = self.encode(self.module(color, **params))
        return [vis]
        
    def __str__(self):
        return f"{self.title}: {self.user}"