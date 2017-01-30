import os
import numpy as np
from skimage import io, feature, color


def safe_remove(array, name):
    try:
        array.remove(name)
    except ValueError:
        pass


def preload():
    files = os.listdir(os.path.curdir + '/data/download/')
    safe_remove(files, '.gitkeep')
    safe_remove(files, '.DS_Store')
    for file in files:
        image = io.imread(os.path.curdir + '/data/download/' + file, as_grey=True)
        edges = np.invert((np.uint8(feature.canny(image, sigma=1) * 256)))
        edges = color.gray2rgb(edges)
        io.imsave(os.path.curdir + '/data/line/' + file + '.tiff', edges)
