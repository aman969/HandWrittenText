from __future__ import division
from __future__ import print_function

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt

import skimage
from skimage.filters import threshold_local, threshold_yen


def preprocessImage(imgPath, imgSize, binary=True):
    """ Pre-processing image for predicting """
    img = cv2.imread(imgPath)
    # Binary
    if binary:
        brightness = 0
        contrast = 50
        img = np.int16(img)
        img = img * (contrast/127+1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        T = threshold_local(img, 11, offset=10, method="gaussian")
        img = (img > T).astype("uint8") * 255

        # Increase line width
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.erode(img, kernel, iterations=1)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create target image and copy sample image into it
    (wt, ht) = imgSize
    (h, w) = img.shape
    fx = w / wt
    fy = h / ht
    f = max(fx, fy)

    # Scaling
    newSize = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    img = cv2.resize(img, newSize)
    target = np.ones([ht, wt]) * 255
    target[0:newSize[1], 0:newSize[0]] = img

    # Transpose
    img = cv2.transpose(target)

    # Normalize using mean stander division
    (m, s) = cv2.meanStdDev(img)
    m = m[0][0]
    s = s[0][0]
    img = img - m
    img = img / s if s > 0 else img

    return img
