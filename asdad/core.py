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

predictor_path = "shape_predictor_68_face_landmarks.dat"

def vec_len(vec):
    return np.sqrt(np.dot(vec, vec))

def features_count():
    return 17

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
sess = tf.Session()
my_head_pose_estimator = CnnHeadPoseEstimator(sess)
#my_head_pose_estimator.load_yaw_variables("tensorflow/head_pose/yaw/cnn_cccdd_30k")
my_head_pose_estimator.load_pitch_variables("send_nudes/asdad/tensorflow/head_pose/pitch/cnn_cccdd_30k.tf")

def get_landmarks(img):
    rects, score, idx = detector.run(img, 1, 1)

    if len(rects) == 0:
        print("Fuck, no faces")
        return [], 0, 0, 0

    rect = rects[0]

    res = np.array([[p.x, p.y] for p in predictor(img, rect).parts()])

    expand = 20
    sz = max(rect.right() - rect.left(), rect.bottom() - rect.top(), 64)
    img = img[rect.left() - expand:rect.left() + sz + expand, rect.top() - expand:rect.top() + sz + expand]
    try:
        pitch = my_head_pose_estimator.return_pitch(img)
    except:
        print("Tensor flow fucked up")
        return res, [[[0.0]]], score, idx
    return res, pitch, score, idx

def get_normalized_face(landmarks):
    nose_to_left = vec_len(landmarks[2] - landmarks[30])
    nose_to_right = vec_len(landmarks[2] - landmarks[14])
    left_to_right = vec_len(landmarks[14] - landmarks[30])
    rotation_ratio = (nose_to_left - nose_to_right) / left_to_right

    chain = landmarks[2:15, :]

    chain -= chain[0]
    basis = np.matrix([chain[12], [-chain[12][1], chain[12][0]]])
    transform = np.linalg.inv(basis)
    #print("Transform matrix:\n", transform)
    chain = chain.astype(float)
    for i in range(1, len(chain)):
        chain[i] = np.dot(transform, chain[i])

    eblo_squared_ratio = vec_len(chain[0] - chain[-1]) / vec_len(chain[2] - chain[-3])
    chain[0] = np.array([eblo_squared_ratio, 0.0])
    chain = np.append(np.array([rotation_ratio, 0.0]), chain)
    return chain

def porn_to_vec(img):
    landmarks, pitch, score, idx = get_landmarks(img)
    if (len(landmarks) == 0):
        return []
    normalized = get_normalized_face(landmarks)
    res = np.append(np.array([pitch[0][0][0], 0.0]), normalized)
    res = np.append(np.array([idx[0], 0.0]), res)
    res = np.append(np.array([score[0], 0.0]), res)
    return res.reshape(features_count(), 2)

def hash_to_string(h):
    res = ""
    for vec in h:
        res += str(vec[0]) + ' ' + str(vec[1]) + '\n'
    return res

def fill_db(folder):
    start = time.time()
    for f in glob.glob(os.path.join(folder, "* (Custom).jpg")):
        print("File: {}".format(f))
        img = io.imread(f)

        str = hash_to_string(porn_to_vec(img))

        if (len(str) == 0):
            continue

        output = open(f + ".hash", "w")
        output.write(str)
        output.close()
    print(time.time() - start)

if (__name__ == '__main__'):
    fill_db("../../Zishy")

