from PIL import Image
import io, base64, requests, json
import numpy as np

class Poster:
    def __init__(self, params):
        self.url = params.AUTH.URL
        self.username = params.AUTH.USERNAME
        self.password = params.AUTH.PASSWORD
        self.params = json.dumps(params)
    
    def encode(self, *args):
        encodes = []
        for i, arg in enumerate(args):
            if type(arg)==np.ndarray:
                if arg.dtype == np.uint8:
                    img = Image.fromarray(arg)
                    f = io.BytesIO()
                    img.save(f, format="PNG")
                    img = f.getvalue()
                    encodes.append(base64.encodebytes(img))
                    
                else:
                    f = io.BytesIO()
                    np.savez_compressed(f, data=arg)
                    val = f.getvalue()
                    encodes.append(base64.encodebytes(val))
                    
            else:
                encodes.append(arg)
        return encodes
    
    def decode(self, *args):
        decodes = []
        for arg in args:
            try:
                img = base64.b64decode(arg)
                f = io.BytesIO()
                f.write(img)
                decodes.append(np.array(Image.open(f)))
                
            except:
                decodes.append(np.load(io.BytesIO(base64.b64decode(arg))))
            
            decodes.append(arg)
        return decodes
        
    def post(self, color, depth=None, coord=None, date_time=None):
        _color, _depth, _coord = self.encode(color, depth, coord)
        response = requests.post(
            self.url,
            data={
                'username': self.username, 'password': self.password,
                'color': _color, 'depth': _depth, 'coord': _coord,
                'params': self.params, 'date_time': date_time
            }
        )
        if response.ok:
            content = response.content.decode('utf-8')
            if "[Error]" in content:
                print(content)
            else:
                vis = self.decode(content)[0][:,:,::-1]
                vis = Image.fromarray(vis)
                return vis
                
        else:
            print("Failed to upload")