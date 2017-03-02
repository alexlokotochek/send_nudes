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
import core
from core import vec_len

folder = "/home/alaktionov/dataset"

# confidence
# eblo_type
# pitch
# yaw
# eblo_square
# chin_points (13)

weights = np.array([0.] + [0.] + [0.05] + [0.05] + [7.] * 15 + [1.] * (core.features_count() - 19))
hashes = []

def find_distance(h1, h2):
    global weights
    res = np.array([vec_len(a) ** 2 for a in (h1 - h2)]) * weights
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
    
def init():
    global hashes
    for f in glob.glob(os.path.join(folder, "* (Custom).jpg.hash")):
        h = load_hash(f)
        hashes.append((f, h))
    print("Hashes loaded:", len(hashes))
    
def find_closest(img):
    global hashes
    print("Current hashes size", len(hashes))
    my_hash = core.get_vector(img)
    if (len(my_hash) == 0):
        print("Fuck")
        return img

    distance = -1.0
    res_hash = None
    ans = None
    for name, h in hashes:
        new_dist = find_distance(my_hash, h)
        if distance < 0 or distance > new_dist:
            distance = new_dist
            ans = name
            res_hash = h
    if ans is None:
        raise Exception('No suitable images found')
    print(ans)
    print(my_hash)
    print(res_hash)
    print(distance)

    return cv2.imread(ans[0:-5], cv2.IMREAD_COLOR)
