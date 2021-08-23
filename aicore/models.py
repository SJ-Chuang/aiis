from django.db import models
from django.conf import settings
from django.utils import timezone
from django.contrib.auth.models import User
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
    
    def encode_img(self, img):
        img = Image.fromarray(img)
        f = io.BytesIO()
        img.save(f, format="PNG")
        content = f.getvalue()
        return base64.encodebytes(content)
    
    def encode_array(self, **kwargs):
        f = io.BytesIO()
        np.savez_compressed(f, **kwargs)
        return base64.encodebytes(f.getvalue())
    
    def decode_img(self, img):
        img = base64.b64decode(img)
        f = io.BytesIO()
        f.write(img)
        return np.array(Image.open(f))
    
    def decode_array(self, array):
        return np.load(io.BytesIO(base64.b64decode(array)))
        
    def publish(self, request):
        arrays = self.decode_array(request.POST.get('arrays', ''))
        color = self.decode_img(arrays['color'])
        depth = self.decode_img(arrays['depth'])
        coord = arrays['coord']
        parameters = json.loads(request.POST.get("parameters"))
        parameters['coord'] = coord
        vis = self.encode_img(self.module(color, **parameters))
        return self.encode_array(vis=vis)
        
    def __str__(self):
        return f"{self.title}: {self.user}"