import picarus_takeout
import time
import base64
import msgpack
import numpy as np
import cv2
import pylab


def fit_homography(p):
    out = cv2.findHomography(np.ascontiguousarray(p[:, 2:]), np.ascontiguousarray(p[:, :2]), cv2.RANSAC, ransacReprojThreshold=70)
    print(out)
    b = project_points(np.ascontiguousarray(p[:, 2:]), out[0]).reshape((-1, 2))
    print(np.hstack([b, p[:, :2]]))
    return out[0]

def project_points(points, h):
    return cv2.perspectiveTransform(points.reshape((1, -1, 2)), h)

def click_points(n, im):
    fig = pylab.figure(1);
    pylab.imshow(im)
 
    def pick(event):
        points.append((event.xdata, event.ydata))
        print('Picked point %d of %d' % (len(points),n))
 
    fig.canvas.mpl_connect('close_event', lambda _: sys.exit(1))
    cid = fig.canvas.mpl_connect('button_press_event', pick)
    points = []
    print("Click %d points" % (n,))
 
    while len(points) < n:
        pylab.waitforbuttonpress()
 
    print "Ok!", points
    return points


def imdecode(data):
    return cv2.imdecode(np.fromstring(data, dtype=np.uint8), 0)

def image_size(data):
    return map(int, imdecode(data).shape)

class ImagePoints(object):

    def __init__(self, verbose=False):
        model = "kYKia3eDrXBhdHRlcm5fc2NhbGXLP/AAAAAAAACmdGhyZXNoFKdvY3RhdmVzAqRuYW1lu3BpY2FydXMuQlJJU0tJbWFnZUZlYXR1cmUyZA=="
        self.model = picarus_takeout.ModelChain(base64.b64decode(model))
        self.verbose = verbose

    def __call__(self, image):
        st = time.time()
        try:
            return self.model.process_binary(image)
        finally:
            if self.verbose:
                print('ImagePoints[%f]' % (time.time() - st,))

class ImageMatch(object):

    def __init__(self, verbose=False):
        model = "kYKia3eDqG1heF9kaXN0eKttaW5faW5saWVycwqtcmVwcm9qX3RocmVzaMtAFAAAAAAAAKRuYW1l2gAkcGljYXJ1cy5JbWFnZUhvbW9ncmFwaHlSYW5zYWNIYW1taW5n"
        self.model = picarus_takeout.ModelChain(base64.b64decode(model))
        self.verbose = verbose

    def __call__(self, p0, p1):
        st = time.time()
        try:
            mat, sz = msgpack.loads(self.model.process_binary(msgpack.dumps([p0, p1])))
            return np.array(mat).reshape(sz)
        finally:
            if self.verbose:
                print('ImageMatch[%f]' % (time.time() - st,))
