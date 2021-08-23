from PIL import Image
import numpy as np
import io, cv2, time, base64, requests, json

class Timer:
    """
    A timer for speed testing.
    Attributes:
        cTime (float): current time.time()
    
    Example
    ::
    timer = Timer
    timer.start()
    <Description 1>
    print(f"The time taken for Description 1 was {timer.stop()}")
    <Description 2>
    print(f"The time taken for Description 2 was {timer.stop()}")
    """
    def __init__(self):
        self.cTime = self.getTime()
    
    def getTime(self):
        return time.time()
    
    def start(self):
        self.cTime = self.getTime()
    
    def stop(self, restart=True):
        deltaT = self.getTime()-self.cTime
        if restart:
            self.cTime = self.getTime()
        return deltaT

class Poster:
    def __init__(self):
        pass
    
    def encode(self, **kwargs):
        for name, value in kwargs.items():
            print(name, value)
        

def encode_img(img):
    img = Image.fromarray(img)
    f = io.BytesIO()
    img.save(f, format="PNG")
    img_bin = f.getvalue()
    return base64.encodebytes(img_bin)

def encode_array(**kwargs):
    f = io.BytesIO()
    np.savez_compressed(f, **kwargs)
    return base64.encodebytes(f.getvalue())

def decode_img(img):
    img = base64.b64decode(img)
    f = io.BytesIO()
    f.write(img)
    return np.array(Image.open(f))

def decode_array(array):
    return np.load(io.BytesIO(base64.b64decode(array)))

"""
Create data for upload.
    color (np.ndarray): a color image.
    depth (np.ndarray): a depth map.
    coord (np.ndarray): 3-d coordinates.
Encode functions:
    encode_img(img): Only color images and depth maps are converted to
        base64 encoding, because PIL.Image supports uint8 format images.
    encode_array(**kwargs): The compressed images and coordinates are
        integrated into one for uploading.
"""
timer = Timer()
timer.start()
color = cv2.imread('data/color.png')
depth = cv2.imread('data/depth.png')
coord = np.load('data/coord.npy')

print(f"It took {timer.stop():.3f} sec for loading.")

arrays = encode_array(
    color=encode_img(color),
    depth=encode_img(depth),
    coord=coord
)

print(f"It took {timer.stop():.3f} sec for encoding.")

"""
Post data to http://140.112.13.4/aicore/upload/.
    data (dict): a dict with username, password and upload data.
***Note***
    Login is necessary before uploading. Login page: http://140.112.13.4/.
"""
response = requests.post(
    "http://140.112.13.4/aicore/upload/",
    data={'username':'aiis', 'password':'aiis', 'arrays': arrays,
        'parameters': json.dumps({
            'format': 'BGR',
        })}
)
print(f"It took {timer.stop():.3f} sec for posting.")

"""
Processing the returned response.
"""
data = response.content.decode('utf-8')
if response.ok:
    if "[Error]" in data:
        print(data)
    else:
        decoded_array = decode_array(data)
        vis = decode_img(decoded_array['vis'])
        cv2.imwrite('vis.png', vis)
        print("Save reulst to vis.png")
else:
    print("Failed to upload")
