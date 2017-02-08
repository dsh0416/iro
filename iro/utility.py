import cv2
import numpy as np


def rgb2yuv(image):
    return cv2.cvtColor(np.ubyte(image * 255), cv2.COLOR_RGB2YUV).astype(np.float) / 255


def yuv2rgb(image):
    return cv2.cvtColor(np.ubyte(image * 255), cv2.COLOR_YUV2RGB).astype(np.float) / 255
