import wearscript
import os
import base64
import cv2
import argparse
import numpy as np

def main():
    out_dir = 'big2glass'
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    def callback(ws, **kw):
        def handler(*data):
            open(out_dir + '/%f-%d-%d.jpg' % (data[3], data[2][0], data[2][1]), 'w').write(base64.b64decode(data[1]))
        ws.subscribe('calibimageb64pt', handler)
        ws.handler_loop()
    wearscript.parse(callback, argparse.ArgumentParser())


if __name__ == '__main__':
    main()
