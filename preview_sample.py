import wearscript
import os
import base64
import cv2
import argparse
import numpy as np

def image_size(data):
    return map(int, cv2.imdecode(np.fromstring(data, dtype=np.uint8), 0).shape)

def main():
    out_dir = 'images'
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    def callback(ws, **kw):
        def image_handler(*data):
            print(data[0])
            height, width = image_size(data[2])
            print((height, width))
            fn = '%s/%.5d_%.5d.jpg' % (out_dir, height, width)
            open(fn, 'w').write(data[2])
        def imageb64_handler(*data):
            data = list(data)
            data[2] = base64.b64decode(data[2])
            image_handler(*data)
        ws.subscribe('image', image_handler)
        ws.subscribe('imageb64', imageb64_handler)
        ws.handler_loop()
    wearscript.parse(callback, argparse.ArgumentParser())


if __name__ == '__main__':
    main()
