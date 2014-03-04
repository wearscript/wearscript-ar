import matplotlib
matplotlib.use('wx')
import cv2
import glob
import os.path
import numpy as np
import picarus_takeout
import base64
import time
import msgpack
import pylab

class ImagePoints(object):
    def __init__(self):
        model = "kYKia3eDrXBhdHRlcm5fc2NhbGXLP/AAAAAAAACmdGhyZXNoFKdvY3RhdmVzAqRuYW1lu3BpY2FydXMuQlJJU0tJbWFnZUZlYXR1cmUyZA=="
        self.model = picarus_takeout.ModelChain(base64.b64decode(model))
    def __call__(self, image):
        st = time.time()
        try:
            return self.model.process_binary(image)
        finally:
            print('ImagePoints[%f]' % (time.time() - st,))

class ImageMatch(object):
    def __init__(self):
        model = "kYKia3eDqG1heF9kaXN0eKttaW5faW5saWVycwqtcmVwcm9qX3RocmVzaMtAFAAAAAAAAKRuYW1l2gAkcGljYXJ1cy5JbWFnZUhvbW9ncmFwaHlSYW5zYWNIYW1taW5n"
        self.model = picarus_takeout.ModelChain(base64.b64decode(model))
    def __call__(self, p0, p1):
        st = time.time()
        try:
            r = msgpack.loads(self.model.process_binary(msgpack.dumps([p0, p1])))
            print(r)
            return np.array(r[0]).reshape((3,3))
        finally:
            print('ImageMatch[%f]' % (time.time() - st,))

def match(base, im):
    pts0 = ImagePoints()(base)
    pts1 = ImagePoints()(im)
    m = ImageMatch()(pts0,pts1)
    return m

def get_points_files():
    files = sorted(glob.glob('*.jpg'))
    for f in files:
        x,y = map(float, f.split('.')[1].split('-')[1:])
        yield x,y,f

def get_matches(xyf):
    xyf = list(xyf)
    base = open(xyf[0][2]).read()
    for x,y,f in xyf:
        m = match(base, open(f).read())
        yield x,y,f,m
        
def click_point(im):
    fig = pylab.figure(1);
    pylab.imshow(im)
    pylab.xlim([700,1500])
    pylab.ylim([900,250])
    point = []
    def pick(event): 
        point.append((event.xdata, event.ydata))
    cid = fig.canvas.mpl_connect('button_press_event', pick)
    print("Click a point")
    while not point: pylab.waitforbuttonpress()
    print "Ok!", point
    return point[0]

def show_xyfm(point_in_0, xyfm):
    x0,y0 = point_in_0
    for i,(x,y,f,m) in enumerate(xyfm):
        pylab.figure(i);
        pylab.imshow(pylab.imread(f))
        xyw = np.dot(m,[x0,y0,1])
        x1,y1 = xyw[:2]/xyw[2]
        pylab.scatter(x1,y1,c='r',marker='+')

def solve_homo(point_in_0, xyfm):
    x0,y0 = point_in_0
    xyfm = list(xyfm)
    def xyuv():
        for i,(x,y,f,m) in enumerate(xyfm):
            xyw = np.dot(m,[x0,y0,1])
            x1,y1 = xyw[:2]/xyw[2]
            yield x,y,x1,y1
            
    xyuv = np.array(list(xyuv()))
    H,_ = cv2.findHomography(xyuv[:,:2].astype('f'),xyuv[:,2:].astype('f'))
    return H

def demo_homo(point_in_0, xyfm, H):
    x0,y0 = point_in_0
    xyfm = list(xyfm)
    def xyuv():
        for i,(x,y,f,m) in enumerate(xyfm):
            xyw = np.dot(m,[x0,y0,1])
            x1,y1 = xyw[:2]/xyw[2]
            yield x,y,x1,y1
    xyuv = np.array(list(xyuv()))
    uvw = np.dot(H, np.vstack((xyuv[:,:2].T,np.ones(xyuv.shape[0]))))
    uv2 = (uvw[:2,:]/uvw[2,:]).T
    for i,(x,y,f,m) in enumerate(xyfm):
        pylab.figure(i)
        pylab.clf()
        pylab.imshow(cv2.imread(f))
        pylab.scatter(xyuv[:,2], xyuv[:,3], marker='+', c='r')
        pylab.scatter(uv2[:,0], uv2[:,1], marker='+', c='g')


# xyfm = list(get_matches(get_points_files()))
# point_in_0 = click_point(imread(xyfm[0][2]))
# H = solve_homo(point_in_0, xyfm)
# demo_homo(point_in_0, xyfm, H)

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

if __name__ == '__main__' and not run_from_ipython():
    # Looking for jpegs in the current directory 
    files = glob.glob('*.jpg')
    print 'Usage: python ../glass2big.py'
    print '  Looks for *.jpg in the current directory. %d found.' % (len(files),)
    print 'Press enter to match images then click a point.' 
    raw_input()
    xyfm = list(get_matches(get_points_files()))
    point_in_0 = click_point(cv2.imread(xyfm[0][2]))
    H = solve_homo(point_in_0, xyfm)
    print 'Done!'
    print(H)
    demo_homo(point_in_0, xyfm, H)
    pylab.show()
