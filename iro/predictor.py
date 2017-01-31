from skimage import io, transform
import numpy as np
from keras.models import load_model
from iro.utility import rgb2hls, hls2rgb


def predict(model_name, file):
    model = load_model("./checkpoint/" + model_name + '.hdf5')
    image = rgb2hls(transform.resize(io.imread(file), (128, 128)))
    result = model.predict(np.array([image]))
    for x in np.nditer(result, op_flags=['readwrite']):
        if x > 1:
            x[...] = 1
        elif x < 0:
            x[...] = 0
    io.imsave('./test.tiff', np.ubyte(hls2rgb(result[0]) * 255))
