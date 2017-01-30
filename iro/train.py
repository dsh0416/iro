from keras.models import Model, load_model
from keras.layers import Input, Convolution2D, MaxPooling2D, UpSampling2D, merge
from keras.callbacks import ModelCheckpoint
from iro.mapper import Generator


def network():
    inputs = Input((256, 256, 3))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)

    up5 = merge([UpSampling2D(size=(2, 2))(conv4), conv3], mode='concat', concat_axis=3)
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up5)
    conv5 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv2], mode='concat', concat_axis=3)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv1], mode='concat', concat_axis=3)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv7)

    conv8 = Convolution2D(3, 1, 1, activation='sigmoid')(conv7)

    model = Model(input=inputs, output=conv8)

    return model


def train():
    try:
        model = load_model('./checkpoint/checkpoint.newest.hdf5')
        print('Network Rescued...')
    except Exception:
        print('Creating Network...')
        model = network()
        print('Building Network...')
        model.compile(optimizer='Nadam', loss='mse')

    print('Loading Data...')
    generator = Generator()
    print('Training...')
    model.fit_generator(generator.next(),
                        samples_per_epoch=32,
                        nb_epoch=40000,
                        nb_worker=1,
                        callbacks=[
                            ModelCheckpoint('./checkpoint/checkpoint.{epoch:02d}.hdf5', verbose=0,
                                            save_best_only=False, save_weights_only=False, mode='auto',
                                            period=100),
                            ModelCheckpoint('./checkpoint/checkpoint.newest.hdf5', verbose=0,
                                            save_best_only=False, save_weights_only=False, mode='auto',
                                            period=1),
                        ])
    print('Training Finished...')
