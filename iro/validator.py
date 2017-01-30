import os
from keras.models import load_model
from iro.preload import safe_remove
import numpy as np
from skimage import transform, color, io


def validate():
    models = os.listdir('./checkpoint')
    safe_remove(models, '.gitkeep')
    safe_remove(models, '.DS_Store')
    image = transform.resize(color.rgb2hsv(io.imread('./data/line/61127521_p0_master1200.jpg.tiff')), (256, 256))
    for model_file in models:
        model = load_model("./checkpoint/" + model_file)
        result = model.predict(np.array([image]))
        for x in np.nditer(result, op_flags=['readwrite']):
            if x > 1:
                x[...] = 1
            elif x < 0:
                x[...] = 0
        io.imsave('./validator/' + model_file + '.tiff', np.ubyte(color.hsv2rgb(result[0]) * 255))
