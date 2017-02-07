import os
import numpy as np
from skimage import io, feature, color


def safe_remove(array, name):
    try:
        array.remove(name)
    except ValueError:
        pass


def preload():
    files = os.listdir('./data/download/')
    safe_remove(files, '.gitkeep')
    safe_remove(files, '.DS_Store')
    for file in files:
        image = io.imread('./data/download/' + file, as_grey=True)
        canny = feature.canny(image, sigma=1)
        edges = np.invert(np.uint8(canny * 255))
        edges = color.gray2rgb(edges)
        io.imsave('./data/line/' + file + '.tiff', edges)
