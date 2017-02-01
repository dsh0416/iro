import os
from keras.models import load_model
from iro.preload import safe_remove
import numpy as np
from skimage import transform, io
from iro.utility import rgb2hls, hls2rgb
from iro.network import GAN


def validate():
    models = os.listdir('./checkpoint')
    safe_remove(models, '.gitkeep')
    safe_remove(models, '.DS_Store')
    image = rgb2hls(transform.resize(io.imread('./data/line/61127521_p0_master1200.jpg.tiff'), (128, 128)))
    gan = GAN()
    generator = gan.generator_network
    generator.load_weights("./checkpoint/generator.new.hdf5")
    result = generator.predict(np.array([image]))
    for x in np.nditer(result, op_flags=['readwrite']):
        if x > 1:
            x[...] = 1
        elif x < 0:
            x[...] = 0
    io.imsave('./validator/' + 'test.tiff', np.ubyte(hls2rgb(result[0]) * 255))
