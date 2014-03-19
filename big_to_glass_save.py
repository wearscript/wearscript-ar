import wearscript
import gevent
import json
import time
import glob
import os
import base64
import cv2
import match
import argparse
import numpy as np

def doit(out_dir):
    immatch, impoints = match.ImageMatch(), match.ImagePoints()
    
    session_glass_pt_features = [(x.rsplit('/')[-1].split('-')[0], map(int, x.rsplit('/')[-1].rsplit('.', 1)[0].split('-')[1:])[::-1], impoints(open(x).read())) for x in glob.glob(out_dir + '/*.jpg')]
    session_glass_pt_features = sorted(session_glass_pt_features)
    session_glass_pt_features_dict = {}
    for session, glass_pt, features in session_glass_pt_features:
        session_glass_pt_features_dict.setdefault(session, []).append((glass_pt[::-1], features))
    pairs = []
    for session, glass_pt_features in session_glass_pt_features_dict.items():
        click_pt = json.load(open('%s/%s-pt.js' % (out_dir, session)))
        match_ind = 5
        for (glass_pt, features) in glass_pt_features[:match_ind] + glass_pt_features[match_ind + 1:]:
            try:
                h = immatch(glass_pt_features[match_ind][1], features)
                image_pt = np.dot(h, [click_pt[0], click_pt[1], 1])
                image_pt = image_pt / image_pt[2]
                pairs.append([glass_pt[0], glass_pt[1], image_pt[0], image_pt[1]])
            except RuntimeError:
                pass
    pairs = np.array(pairs, dtype=np.float32)
    print(pairs)
    H = match.fit_homography(pairs, thresh=15)
    #H, inliers = cv2.findHomography(np.ascontiguousarray(pairs[:, 2:]), np.ascontiguousarray(pairs[:, :2]))
    print(H)
    print(H.ravel().tolist())
    return


def main():
    out_dir = 'big2glass_test'
    try:
        os.makedirs(out_dir)
    except OSError:
        pass
    doit(out_dir)
    return
    images = []  # (time, x, y, data)
    def callback(ws, **kw):
        def handler(*data):
            image_data = base64.b64decode(data[1])
            open(out_dir + '/%f-%d-%d.jpg' % (data[3], data[2][0], data[2][1]), 'w').write(image_data)
            images.append([data[3], data[2][0],
                           data[2][1], image_data])
            if len(images) == 6:
                ws.publish('annotationimagepoints', 'calibpt:%f' % data[3], data[1], 1, 'Click the point you looked at')

        def calib_pt_handler(channel, pts):
            print(pts)
            open(out_dir + '/%s-pt.js' % channel.split(':')[-1], 'w').write(json.dumps(pts[0]))
        def test():
            time.sleep(.1)
            image_data = base64.b64encode(open(glob.glob(out_dir + '/*.jpg')[0]).read())
            ws.publish('annotationimagepoints', 'calibpt', image_data, 1, 'Click the point you looked at')
        #gevent.spawn(test)
            
        ws.subscribe('calibimageb64pt', handler)
        ws.subscribe('calibpt', calib_pt_handler)
        ws.handler_loop()
    wearscript.parse(callback, argparse.ArgumentParser())


if __name__ == '__main__':
    main()
