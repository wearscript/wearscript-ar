import wearscript
import base64
import os
import random
import msgpack
import argparse
import time
import copy
import picarus_takeout
import numpy as np
from match import ImagePoints, ImageMatch

def main():
    image_points = ImagePoints()
    image_match = ImageMatch()
    match_points = {}

    ar_model = picarus_takeout.ModelChain(msgpack.dumps([{'kw':{}, 'name': 'picarus.ARMarkerDetector'}]))
    print()
    
    def callback(ws, **kw):

        def image_handler(*data):
            groupDevice = '' # data[0].split(':', 1)[1]
            print('Image[%s]' % data[0])
            if groupDevice in match_points:
                st = time.time()
                pts = image_points(data[2])
                print('Points[%f]' % (time.time() - st))
                st = time.time()
                h = image_match(match_points[groupDevice], pts)
                print('Match[%f]' % (time.time() - st))
                print('Sending')
                ws.publish('warph:' + groupDevice, h.ravel().tolist())
            tags, tag_size = msgpack.loads(ar_model.process_binary(data[2]))
            ws.publish('warptags', np.array(tags).reshape(tag_size).tolist())

        def warp_sample_handler(*data):
            print('Warp Sample')
            match_points[data[1]] = image_points(data[2])

        ws.subscribe('image', image_handler)
        ws.subscribe('warpsample', warp_sample_handler)
        ws.handler_loop()
    wearscript.parse(callback, argparse.ArgumentParser())


if __name__ == '__main__':
    main()
