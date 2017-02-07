from keras.models import Sequential, Model
from keras.layers import Convolution2D, Flatten, Dense
from keras.layers import Input, MaxPooling2D, UpSampling2D, merge
from keras.callbacks import EarlyStopping
from iro.data import Data


class Generator:
    def __init__(self):
        inputs = Input((128, 128, 3))
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
        self.network = Model(input=inputs, output=conv8)
        self.network.compile(
            optimizer='RMSprop',
            loss='mse',
        )


class Discriminator:
    def __init__(self):
        self.network = Sequential([
            Convolution2D(128, 2, 2, activation='relu', border_mode='same', input_shape=(128, 128, 3)),
            Convolution2D(128, 3, 3, activation='relu', border_mode='same'),
            MaxPooling2D(pool_size=(2,2)),
            Convolution2D(64, 2, 2, activation='relu', border_mode='same'),
            Convolution2D(64, 3, 3, activation='relu', border_mode='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Convolution2D(32, 2, 2, activation='relu', border_mode='same'),
            Convolution2D(32, 3, 3, activation='relu', border_mode='same'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(1),
        ])
        self.network.compile(
            optimizer='RMSprop',
            loss='mse',
        )


class GAN:
    def __init__(self):
        self.data = None
        self.generator_network = Generator().network
        self.discriminator_network = Discriminator().network

        self.gan_network = Sequential([
            self.generator_network,
            self.discriminator_network,
        ], name='GAN')

        self.gan_network.compile(
            optimizer='RMSprop',
            loss='mse',
        )

    def load_data(self):
        self.data = Data()

    def load_weights(self,
                     generator_weights='./checkpoint/generator.new.hdf5',
                     discriminator_weights='./checkpoint/discriminator.new.hdf5',
                     gan_weights='./checkpoint/gan.new.hdf5'):
        self.generator_network.load_weights(generator_weights)
        self.discriminator_network.load_weights(discriminator_weights)
        self.gan_network.load_weights(gan_weights)

    def train(self, batch_size=8, nb_epoch=1, samples=128, nb_worker=1):
        while True:
            print('Training Discriminator')
            self.discriminator_network.trainable = True
            self.discriminator_network.compile(
                optimizer='RMSprop',
                loss='mse',
            )

            self.discriminator_network.fit_generator(
                self.data.discriminator_next(self.generator_network, batch_size=batch_size),
                samples_per_epoch=samples,
                nb_epoch=nb_epoch,
                nb_worker=nb_worker)

            print('Training Generator')
            self.discriminator_network.trainable = False
            self.discriminator_network.compile(
                optimizer='RMSprop',
                loss='mse',
            )

            self.gan_network.fit_generator(
                self.data.gan_next(batch_size=batch_size),
                samples_per_epoch=samples,
                nb_epoch=nb_epoch,
                nb_worker=nb_worker)

            self.discriminator_network.save_weights('./checkpoint/discriminator.new.hdf5')
            self.generator_network.save_weights('./checkpoint/generator.new.hdf5')
            self.gan_network.save_weights('./checkpoint/gan.new.hdf5')
