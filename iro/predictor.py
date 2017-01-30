from skimage import io, color, transform
import numpy as np
from keras.models import load_model


def predict(model_name, file):
    model = load_model("./checkpoint/" + model_name + '.hdf5')
    image = transform.resize(color.rgb2hsv(io.imread(file)), (256, 256))
    result = model.predict(np.array([image]))
    for x in np.nditer(result, op_flags=['readwrite']):
        if x > 1:
            x[...] = 1
        elif x < 0:
            x[...] = 0
    io.imsave('./test.tiff', np.ubyte(color.hsv2rgb(result[0]) * 255))
