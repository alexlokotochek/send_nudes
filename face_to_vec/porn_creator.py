import PIL.Image
from io import StringIO
from matplotlib import pyplot as plt
import numpy as np

from skimage import io
import cv2
import dlib
import numpy
import sys
import api

"""
params
usage: 
    im1 = io.imread('./ava.jpg')
    im2 = io.imread('./porn.jpg')
    res = get_porn(im1, im2)
im format: np.ndarray with shape width, height, RGB
"""

PREDICTOR_PATH = "./shape_predictor_68_face_landmarks.dat"
FEATHER_AMOUNT = 17
COLOUR_CORRECT_BLUR_FRAC = 0.6
KERNEL_FACTOR = 2

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

ALIGN_POINTS = (LEFT_BROW_POINTS + 
                RIGHT_EYE_POINTS + 
                LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + 
                NOSE_POINTS + 
                MOUTH_POINTS)

OVERLAY_POINTS = [
    LEFT_EYE_POINTS,
    RIGHT_EYE_POINTS, 
    NOSE_POINTS, 
    MOUTH_POINTS,
    [24, 19, 68] 
]

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
        print('Too many faces')
        return None
    if len(rects) == 0:
        print('No faces')
        return None
    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)
    for i, group in enumerate(OVERLAY_POINTS):
        this_landmarks = landmarks[group]
        # this_landmarks = 2*this_landmarks + this_landmarks[:, ::-1]
        draw_convex_hull(im,
                         this_landmarks,
                         color=1)
    im = numpy.array([im, im, im]).transpose((1, 2, 0))
    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    # sum ||s*R*p1,i + T - p2,i||^2 -> min
    points1 = points1.astype(numpy.float64)
    points2 = points2.astype(numpy.float64)
    c1 = numpy.mean(points1, axis=0)
    c2 = numpy.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = numpy.std(points1)
    s2 = numpy.std(points2)
    points1 /= s1
    points2 /= s2
    U, S, Vt = numpy.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return numpy.vstack([numpy.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         numpy.matrix([0., 0., 1.])])


def resize_im(im):
    h, w, _ = im.shape
    koeff_ratio = 1
    if (h > w):
        koeff_ratio = 1024. / h
    else:
        koeff_ratio = 680. / w

    new_h = h * koeff_ratio
    new_w = w * koeff_ratio
    antialias_koeff = min(koeff_ratio, 1.)
    im = cv2.resize(im, (int(new_w),int(new_h)), 
                    fx=koeff_ratio,
                    fy=koeff_ratio,
                    interpolation = cv2.INTER_AREA)
    return im


def read_im_and_landmarks(im):
    s = get_landmarks(im)
    return s


def warp_im(im, M, dshape):
    output_im = numpy.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * numpy.linalg.norm(
                              numpy.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
                              numpy.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    # /0 errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)
    return (im2.astype(numpy.float64) * im1_blur.astype(numpy.float64) /
                                                im2_blur.astype(numpy.float64))

def add_forehead_point(landmarks):
    corner_points = [24, 19, 27] # 27 is upper nose point
    corner_points = -landmarks[:,0][corner_points], -landmarks[:,1][corner_points]
    pt0 = np.array([int(corner_points[0][0]), int(corner_points[1][0])])
    pt1 = np.array([int(corner_points[0][1]), int(corner_points[1][1])])
    pt = np.array([int(corner_points[0][2]), int(corner_points[1][2])])
    line_pt = pt1
    line_vec = pt1 - pt0
    pt_proj = np.dot(pt - line_pt, line_vec) * line_vec / np.dot(line_vec, line_vec)
    pt_norm = pt - line_pt - pt_proj
    pt_sym = pt - 2*pt_norm
    mid = (pt0 + pt1)/2.
    pt_sym += (-pt + mid)*0.2
    return np.vstack([landmarks ,[-pt_sym[0], -pt_sym[1]]])

def get_porn(im1, im2):
    landmarks1 = read_im_and_landmarks(im1)
    landmarks1 = add_forehead_point(landmarks1).astype(int)

    kernel = np.ones((KERNEL_FACTOR, KERNEL_FACTOR),np.float32)/(KERNEL_FACTOR**2)
    im1 = cv2.filter2D(im1, -1, kernel)

    landmarks2 = read_im_and_landmarks(im2)
    landmarks2 = add_forehead_point(landmarks2).astype(int)

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])

    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = numpy.max([get_face_mask(im1, landmarks1), warped_mask],
                              axis=0)

    warped_im2 = warp_im(im2, M, im1.shape)
    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)

    output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    #output_im = resize_im(output_im)
    return output_im
    # cv2.imwrite('output.jpg', output_im)

if (__name__ == '__main__'):
    api.init()
    im = cv2.imread('./ava.jpg', cv2.IMREAD_COLOR)
    res = get_porn(api.find_closest(im), im)
    cv2.imwrite('res.jpg', res)
