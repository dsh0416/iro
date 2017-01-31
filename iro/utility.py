import cv2
import numpy as np


def rgb2hls(image):
    return cv2.cvtColor(np.ubyte(image * 255), cv2.COLOR_RGB2HLS).astype(np.float) / 255


def hls2rgb(image):
    return cv2.cvtColor(np.ubyte(image * 255), cv2.COLOR_HLS2RGB).astype(np.float) / 255
