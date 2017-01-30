import os, random
from iro.preload import safe_remove
import numpy as np
from skimage import io, color, transform, util


def mapper():
    files = os.listdir(os.path.curdir + '/data/download/')
    safe_remove(files, '.gitkeep')
    safe_remove(files, '.DS_Store')
    values = []
    for file in files:
        try:
            image = io.imread(os.path.curdir + '/data/download/' + file)
            image = transform.resize(image, (256, 256))
            line = io.imread(os.path.curdir + '/data/line/' + file + '.tiff')
            line = transform.resize(line, (256, 256))
            if image.shape == (256, 256, 3) and line.shape == (256, 256, 3):
                values.append((line,
                               image))
        except IndexError:
            pass
    return values


class Generator:
    def __init__(self):
        self.cached = mapper()

    def __iter__(self):
        return self

    def size(self):
        return len(self.cached)

    def next(self):
        count = 0
        inputs = []
        outputs = []
        while True:
            for image in self.cached:
                angle = random.randrange(0, 3) * 90
                inputs.append(transform.rotate(color.rgb2hsv(util.random_noise(image[0])), angle))
                outputs.append(transform.rotate(color.rgb2hsv(image[1]), angle))
                count += 1
                if count >= 30:
                    count = 0
                    yield np.array(inputs), np.array(outputs)
                    inputs = []
                    outputs = []
