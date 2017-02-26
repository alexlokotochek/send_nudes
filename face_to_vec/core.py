#!/usr/bin/python

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

def vec_len(vec):
    return np.sqrt(np.dot(vec, vec))

def features_count():
    # confidence
    # eblo_type
    # pitch
    # yaw
    # landmarks (68)
    return 72

MODELS_DIR = os.path.dirname(__file__)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat"))
sess = tf.Session()
rotation_estimator = CnnHeadPoseEstimator(sess)
rotation_estimator.load_yaw_variables(os.path.join(MODELS_DIR, "tensorflow/head_pose/yaw/cnn_cccdd_30k"))
rotation_estimator.load_pitch_variables(os.path.join(MODELS_DIR, "tensorflow/head_pose/pitch/cnn_cccdd_30k.tf"))

def get_vector(img):
    rects, score, idx = detector.run(img, 1, 1)

    if len(rects) == 0:
        print("Fuck, no faces")
        return []

    rect = rects[0]
    # https://matthewearl.github.io/2015/07/28/switching-eds-with-python/
    landmarks = np.array([[p.x, p.y] for p in predictor(img, rect).parts()])
 
    # Use tensorflow to get yaw and pitch
    expand = 20
    sz = max(rect.right() - rect.left(), rect.bottom() - rect.top(), 64)
    img = img[rect.left() - expand:rect.left() + sz + expand, rect.top() - expand:rect.top() + sz + expand]
    try:
        pitch = rotation_estimator.return_pitch(img)
        yaw = rotation_estimator.return_yaw(img)
    except:
        print("Tensor flow fucked up")
        return []
    landmarks = landmarks.astype(float)
    landmarks -= landmarks[30]
    basis = np.matrix([landmarks[8], landmarks[2]])
    transform = np.linalg.inv(basis)
    for i in range(0, len(landmarks)):
        landmarks[i] = np.dot(transform, landmarks[i])
    #chin = landmarks[2:15, :]
    #chin -= chin[0]
    #basis = np.matrix([chin[12], [-chin[12][1], chin[12][0]]])
    #transform = np.linalg.inv(basis)
    #chin = chin.astype(float)
    #for i in range(1, len(chin)):
    #    chin[i] = np.dot(transform, chin[i])
    
    #eblo_square_ratio = vec_len(chin[0] - chin[-1]) / vec_len(chin[2] - chin[-3])
    
    features = np.array([score[0], 0, idx[0], 0, pitch, 0, yaw, 0])
    res = np.append(features, landmarks)
    
    return res.reshape(features_count(), 2)

def vec_to_string(h):
    res = ""
    for vec in h:
        res += str(vec[0]) + ' ' + str(vec[1]) + '\n'
    return res

def fill_db(folder):
    for f in glob.glob(os.path.join(folder, "* (Custom).jpg")):
        print("File:", f)
        img = io.imread(f)

        str = vec_to_string(get_vector(img))

        if (len(str) == 0):
            continue

        output = open(f + ".hash", "w")
        output.write(str)
        output.close()

if (__name__ == '__main__'):
    fill_db("../../dataset")
