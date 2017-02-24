import sys
import os
import dlib
import glob
import tensorflow as tf
import cv2
import dlib
from deepgaze.head_pose_estimation import CnnHeadPoseEstimator
from skimage import draw
from skimage import io
import numpy as np
import time
import send_nudes.asdad.core as core
from send_nudes.asdad.core import vec_len

folder = "/home/ubuntu/Zishy"


hashes = []

weights = np.array([0.] + [0.] + [1.] + [1.] + [1.] * (core.features_count() - 4))

def find_distance(h1, h2):
    global weights
    res = np.array([vec_len(a) for a in (h1 - h2)]) * weights
    if (h1[1][0] ** 2 < 1e-6):
        res[1] = 0
    return np.dot(res, res)

def load_hash(name):
    with open(name, "r") as f:
        vec = []
        for i, line in enumerate(f):
            if len(line) == 0:
                continue
            try:
                x, y = line.split(' ')
                x, y = float(x), float(y)
                vec.append(np.array([x, y]))
            except:
                print(f)
    res = np.array(vec).reshape(core.features_count(), 2)
    return res

def find_closest(img):
    global hashes
    my_hash = core.porn_to_vec(img)

    if (len(my_hash) == 0):
        print("Fuck")
        return img

    distance = -1.0
    res_hash = None
    ans = ""
    for name, h in hashes:
        new_dist = find_distance(my_hash, h)
        if (distance < 0):
            distance = new_dist
            ans = name
            res_hash = h
        elif (distance > new_dist):
            distance = new_dist
            ans = name
            res_hash = h
    print(ans)
    print(my_hash)
    print(res_hash)
    # Debug
    for vec in res_hash - my_hash:
        print(vec_len(vec))

    return cv2.imread(ans[0:-5], cv2.IMREAD_COLOR)



def init():
    global hashes
    for f in glob.glob(os.path.join(folder, "* (Custom).jpg.hash")):
        h = load_hash(f)
        hashes.append((f, h))

