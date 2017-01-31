from keras.models import Sequential, Model
from keras.layers import Convolution2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Reshape
from keras.callbacks import ReduceLROnPlateau
from iro.data import Data
import numpy as np


class GAN:
    def __init__(self):
        self.generator_network = Sequential([
            Convolution2D(32, 4, 4,
                          input_shape=(128, 128, 3),
                          activation='relu',
                          border_mode='same',
                          subsample=(2, 2)),
            BatchNormalization(),
            Convolution2D(32, 3, 3,
                          activation='relu',
                          border_mode='same',
                          subsample=(1, 1)),
            BatchNormalization(),
            Convolution2D(64, 4, 4,
                          activation='relu',
                          border_mode='same',
                          subsample=(2, 2)),
            BatchNormalization(),
            Convolution2D(64, 3, 3,
                          activation='relu',
                          border_mode='same',
                          subsample=(1, 1)),
            BatchNormalization(),
            Convolution2D(128, 4, 4,
                          activation='relu',
                          border_mode='same',
                          subsample=(2, 2)),
            BatchNormalization(),
            Convolution2D(128, 3, 3,
                          activation='relu',
                          border_mode='same',
                          subsample=(1, 1)),
            BatchNormalization(),
            MaxPooling2D((16, 16)),
            Dense(128 * 128 * 3, activation='sigmoid'),
            Reshape((128, 128, 3)),
        ], name='Generator')

        self.discriminator_network = Sequential([
            Convolution2D(32, 4, 4,
                          input_shape=(128, 128, 3),
                          activation='relu',
                          border_mode='same',
                          subsample=(2, 2)),
            BatchNormalization(),
            Convolution2D(32, 3, 3,
                          activation='relu',
                          border_mode='same',
                          subsample=(1, 1)),
            BatchNormalization(),
            Convolution2D(64, 4, 4,
                          activation='relu',
                          border_mode='same',
                          subsample=(2, 2)),
            BatchNormalization(),
            Convolution2D(64, 3, 3,
                          activation='relu',
                          border_mode='same',
                          subsample=(1, 1)),
            BatchNormalization(),
            Convolution2D(128, 4, 4,
                          activation='relu',
                          border_mode='same',
                          subsample=(2, 2)),
            BatchNormalization(),
            Convolution2D(128, 3, 3,
                          activation='relu',
                          border_mode='same',
                          subsample=(1, 1)),
            BatchNormalization(),
            MaxPooling2D((16, 16)),
            Flatten(),  # 128 * 128 * 3
            Dense(128 * 128 * 3, activation='relu'),
            Dense(256, activation='relu'),
            Dense(1, activation='sigmoid'),
        ], name='Discriminator')

        self.gan_network = Sequential([
            self.generator_network,
            self.discriminator_network,
        ], name='GAN')

        self.generator_network.compile(
            optimizer='Nadam',
            loss='mse',
        )

        # Magic error caused by TensorFlow if prediction not called at first,
        # Fix with magics mentioned on [http://tsuwabuki.hatenablog.com/entry/2016/10/17/150033]
        x = np.zeros((1, 128, 128, 3))
        self.generator_network.predict(x)

        self.discriminator_network.compile(
            optimizer='Nadam',
            loss='mse',
        )
        self.gan_network.compile(
            optimizer='Nadam',
            loss='mse',
        )

        self.gan_network.summary()

        self.data = Data()

    def load_weights(self,
                     generator_weights='./checkpoint/discriminator.new.hdf5',
                     discriminator_weights='./checkpoint/generator.new.hdf5',
                     gan_weights='./checkpoint/gan.new.hdf5'):
        self.generator_network.load_weights(generator_weights)
        self.discriminator_network.load_weights(discriminator_weights)
        self.gan_network.load_weights(gan_weights)

    def train(self, batch_size=8, nb_epoch=4, samples=128, nb_worker=1):
        while True:
            print('Training Discriminator')
            self.discriminator_network.trainable = True

            self.discriminator_network.fit_generator(
                self.data.discriminator_next(self.generator_network, batch_size=batch_size),
                validation_data=self.data.discriminator_next(self.generator_network, batch_size=batch_size),
                samples_per_epoch=samples,
                nb_val_samples=samples,
                nb_epoch=nb_epoch,
                nb_worker=nb_worker,
                callbacks=[ReduceLROnPlateau()],
            )

            print('Training Generator')
            self.discriminator_network.trainable = False
            self.gan_network.fit_generator(
                self.data.gan_next(batch_size=batch_size),
                validation_data=self.data.gan_next(batch_size=batch_size),
                samples_per_epoch=samples,
                nb_val_samples=samples,
                nb_epoch=nb_epoch,
                nb_worker=nb_worker,
                callbacks=[ReduceLROnPlateau()],
            )

            self.discriminator_network.save_weights('./checkpoint/discriminator.new.hdf5')
            self.generator_network.save_weights('./checkpoint/generator.new.hdf5')
            self.gan_network.save_weights('./checkpoint/gan.new.hdf5')
