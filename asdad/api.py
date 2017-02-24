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
import pidor
from pidor import vec_len

folder = "../../MetArt"

weights = np.array([100.] + [2.] + [1.]*12)

def find_distance(h1, h2):
    global weights
    res = np.array([vec_len(a) for a in (h1 - h2)]) * weights
    if (h1[1][0] ** 2 < 1e-6):
        res[1] = np.array([0, 0])
    return np.dot(res, res)

def load_hash(name):
    with open(name, "r") as f:
        vec = []
        for i, line in enumerate(f):
            if len(line) == 0:
                continue
            if i >= 2:
                x, y = line.split(' ')
                x, y = float(x), float(y)
            else:
                x = float(line)
                y = 0.
            vec.append(np.array([x, y]))
    eblo_schjat = vec_len(vec[-1]) / vec_len(vec[3] - vec[-3])
    vec = np.append(np.array([eblo_schjat, 0]), vec)
    res = np.array(vec).reshape(14, 2)
    return res

def find_closest(img):
    my_hash = pidor.porn_to_vec(img)

    if (len(my_hash) == 0):
        print("Fuck")
        return img

    distance = -1.0
    res_hash = None
    ans = ""
    for f in glob.glob(os.path.join(folder, "* (Custom).jpg.hash")):
        h = load_hash(f)
        new_dist = find_distance(my_hash, h)
        if (distance < 0):
            distance = new_dist
            ans = f
            res_hash = h
        elif (distance > new_dist):
            distance = new_dist
            ans = f
            res_hash = h
    print(ans)
    print(my_hash)
    print(res_hash)
    # Debug
    for vec in res_hash - my_hash:
        print(vec_len(vec))

    return cv2.imread(ans[0:-5], cv2.IMREAD_COLOR)