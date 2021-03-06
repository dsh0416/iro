import random
import os
from iro.preload import safe_remove
import numpy as np
from skimage import io, transform, util
from iro.utility import rgb2yuv


class Data:
    def __init__(self):
        files = os.listdir('./data/download/')
        safe_remove(files, '.gitkeep')
        safe_remove(files, '.DS_Store')
        values = []
        for file in files:
            try:
                image = io.imread('./data/download/' + file)
                image = transform.resize(image, (128, 128))
                line = io.imread('./data/line/' + file + '.tiff')
                line = transform.resize(line, (128, 128))
                if image.shape == (128, 128, 3) and line.shape == (128, 128, 3):
                    values.append((line,
                                   image))
            except IndexError:
                pass
        self.data = values

    def gan_next(self, batch_size=1):
        count = 0
        inputs = []
        outputs = []
        while True:
            for image in self.data:
                angle = random.randrange(0, 3) * 90
                inputs.append(rgb2yuv(transform.rotate(util.random_noise(image[0]), angle)))
                outputs.append([0.0])
                count += 1
                if count >= batch_size:
                    count = 0
                    yield np.array(inputs), np.array(outputs)
                    inputs = []
                    outputs = []

    def discriminator_next(self, generator, batch_size=1):
        count = 0
        inputs = []
        outputs = []
        while True:
            for image in self.data:
                angle = random.randrange(0, 3) * 90
                output = random.randrange(0, 2)
                if output == 1:
                    # Get a fake image
                    inputs.append(transform.rotate(generator.predict(np.array([image[0]]))[0], angle))
                else:
                    # Get a real image
                    inputs.append(rgb2yuv(transform.rotate(image[1], angle)))
                outputs.append([output * 1.0])
                count += 1
                if count >= batch_size:
                    count = 0
                    yield np.array(inputs), np.array(outputs)
                    inputs = []
                    outputs = []
