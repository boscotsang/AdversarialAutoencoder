#!/usr/bin/env python
# encoding: utf-8

"""
@author: BoscoTsang
@date: 16-11-14
"""

import functools
import numpy as np
import matplotlib.pyplot as plt
import theano
import keras.backend as K
from keras.models import Model
from keras.layers import Dense, Input, merge, BatchNormalization, Dropout, Activation
from keras.regularizers import l2
from keras.utils.np_utils import to_categorical

theano.config.dnn.enabled = False


class BaseAdversarialAutoencoder(object):
    def __init__(self, input_shape, encoding_laye=None, decoing_layer=None, disc_layer=None, y_dim=30, z_dim=5,
                 lam=0.0001, optimizer='rmsprop', verbose=0):
        self.input_shape = input_shape
        self.encoding_layer = encoding_laye
        self.decoding_layer = decoing_layer
        self.disc_layer = disc_layer
        self.y_dim = y_dim
        self.z_dim = z_dim
        self.lam = lam
        self.optimizer = optimizer
        self.verbose = verbose

    def _build_generator(self, dropout=None, batchnormalization=False):
        self.x_in = Input((self.input_shape,))
        if dropout is None:
            self.x_e = self.x_in
        else:
            self.x_e = Dropout(dropout)(self.x_in)
        for i, neurons in enumerate(self.encoding_layer):
            self.x_e = Dense(neurons, W_regularizer=l2(self.lam))(self.x_e)
            if batchnormalization:
                self.x_e = BatchNormalization(mode=2)(self.x_e)
            self.x_e = Activation('relu')(self.x_e)
        self.y = Dense(self.y_dim, W_regularizer=l2(self.lam))(self.x_e)
        if batchnormalization:
            self.y = BatchNormalization(mode=2)(self.y)
        self.y = Activation('softmax')(self.y)
        self.z = Dense(self.z_dim, W_regularizer=l2(self.lam))(self.x_e)
        if batchnormalization:
            self.z = BatchNormalization(mode=2)(self.z)
        self.z = Activation('linear')(self.z)
        self.latent_code = merge((self.y, self.z), mode='concat', concat_axis=1)

    def _build_decoder(self):
        for i, neurons in enumerate(self.decoding_layer):
            if 0 == i:
                self.x_r = Dense(neurons, activation='relu', input_shape=(self.y_dim,))(self.latent_code)
            else:
                self.x_r = Dense(neurons, activation='relu')(self.x_r)
        self.x_r = Dense(self.input_shape, activation='sigmoid', W_regularizer=l2(self.lam))(self.x_r)
        self.decode = K.function([self.latent_code], self.x_r)
        self.autoencoder = Model(self.x_in, self.x_r)
        self.autoencoder.compile(self.optimizer, 'mse')
        if 1 == self.verbose:
            print "Autoencoder Summary"
            self.autoencoder.summary()

    def _build_discriminator(self):
        self.latent_code_input = Input((self.y_dim + self.z_dim,))
        self.h = self.latent_code_input
        for i, neurons in enumerate(self.disc_layer):
            self.h = Dense(neurons, activation='relu', W_regularizer=l2(self.lam))(self.h)
        self.d_out = Dense(1, activation='sigmoid', W_regularizer=l2(self.lam))(self.h)
        self.discriminator_z = Model(self.latent_code_input, self.d_out)
        self.discriminator_z.compile(self.optimizer, 'binary_crossentropy')
        if 1 == self.verbose:
            print "Discriminator z Summary"
            self.discriminator_z.summary()
        for l in self.discriminator_z.layers:
            l.trainable = False
        self.x_hat = Input((self.input_shape,))
        self.h_hat = self.discriminator_z(self.generator(self.x_hat))
        self.discriminator_zx = Model(self.x_hat, self.h_hat)
        self.discriminator_zx.compile(self.optimizer, 'binary_crossentropy')
        if 1 == self.verbose:
            print "Discriminator zx Summary"
            self.discriminator_zx.summary()


class UnsupervisedCluster(BaseAdversarialAutoencoder):
    def __init__(self, input_shape, encoding_layer=None, decoding_layer=None, disc_layer=None, n_clusters=30, z_dim=5,
                 lam=0.0001, optimizer='rmsprop', dropout=0.2, batchnormalization=True, verbose=0, z_sample_func=None):
        super(UnsupervisedCluster, self).__init__(input_shape, encoding_layer, decoding_layer, disc_layer,
                                                  n_clusters, z_dim, lam, optimizer, verbose)
        self.z_sample_func = z_sample_func if z_sample_func is not None else functools.partial(np.random.normal,
                                                                                               loc=0.,
                                                                                               scale=1.0)
        self._build_generator(dropout, batchnormalization)
        self._build_decoder()
        self._build_discriminator()

    def _build_generator(self, dropout=None, batchnormalization=False):
        super(UnsupervisedCluster, self)._build_generator(dropout, batchnormalization)
        self.generator_z = Model(self.x_in, self.z)
        if 1 == self.verbose:
            print "Generator_y Summary"
            self.generator_y.summary()
            print "Generator_z Summary"
            self.generator_z.summary()

    def _build_decoder(self):
        super(UnsupervisedCluster, self)._build_decoder()

    def _build_discriminator(self):
        self.y_input = Input((self.y_dim,))
        self.h_y = self.y_input
        for i, neurons in enumerate(self.disc_layer):
            self.h_y = Dense(neurons, activation='relu', W_regularizer=l2(self.lam))(self.h_y)
        self.d_out_y = Dense(1, activation='sigmoid', W_regularizer=l2(self.lam))(self.h_y)
        self.discriminator_y = Model(self.y_input, self.d_out_y)
        self.discriminator_y.compile(self.optimizer, 'binary_crossentropy')
        if 1 == self.verbose:
            print "Discriminator_y Summary"
            print self.discriminator_y.summary()
        for l in self.discriminator_y.layers:
            l.trainable = False
        self.x_hat = Input((self.input_shape,))
        self.h_y_hat = self.discriminator_y(self.generator_y(self.x_hat))
        self.discriminator_yx = Model(self.x_hat, self.h_y_hat)
        self.discriminator_yx.compile(self.optimizer, 'binary_crossentropy')
        if 1 == self.verbose:
            print "Discriminator_yx Summary"
            self.discriminator_yx.summary()

        self.z_input = Input((self.z_dim,))
        self.h_z = self.z_input
        for i, neurons in enumerate(self.disc_layer):
            self.h_z = Dense(neurons, activation='relu', W_regularizer=l2(self.lam))(self.h_z)
        self.d_out_z = Dense(1, activation='sigmoid', W_regularizer=l2(self.lam))(self.h_z)
        self.discriminator_z = Model(self.z_input, self.d_out_z)
        self.discriminator_z.compile(self.optimizer, 'binary_crossentropy')
        if 1 == self.verbose:
            print "Discriminator_z Summary"
            self.discriminator_z.summary()
        for l in self.discriminator_z.layers:
            l.trainable = False
        self.h_z_hat = self.discriminator_z(self.generator_z(self.x_hat))
        self.discriminator_zx = Model(self.x_hat, self.h_z_hat)
        self.discriminator_zx.compile(self.optimizer, 'binary_crossentropy')
        if 1 == self.verbose:
            print "Discriminator_zx Summary"
            self.discriminator_zx.summary()

    def fit(self, X, batch_size=100, nb_epoch=10):
        ae_loss = np.zeros(X.shape[0] / batch_size)
        disc_loss = np.zeros(X.shape[0] / batch_size)
        adv_loss = np.zeros(X.shape[0] / batch_size)
        for epoch in xrange(nb_epoch):
            np.random.shuffle(X)
            for batch in xrange(X.shape[0] / batch_size):
                X_batch = X[batch * batch_size:(batch + 1) * batch_size, :]
                y_generated = to_categorical(np.random.randint(0, self.y_dim, size=X_batch.shape[0]), self.y_dim)
                z_generated = self.z_sample_func(size=(X_batch.shape[0], self.z_dim)).astype(theano.config.floatX)
                ae_loss[batch] = self.autoencoder.train_on_batch(X_batch, X_batch)
                disc_y_x = np.concatenate((self.generator_y.predict(X_batch), y_generated))
                disc_z_x = np.concatenate((self.generator_z.predict(X_batch), z_generated))
                disc_labels = np.concatenate((np.ones(X_batch.shape[0]), np.zeros(X_batch.shape[0])))
                disc_loss[batch] = self.discriminator_y.train_on_batch(disc_y_x, disc_labels) \
                                   + self.discriminator_z.train_on_batch(disc_z_x, disc_labels)
                adv_loss[batch] = self.discriminator_yx.train_on_batch(X_batch, np.zeros(X_batch.shape[0])) + \
                                  self.discriminator_zx.train_on_batch(X_batch, np.zeros(X_batch.shape[0]))
            print "AE LOSS: {}, DISC LOSS: {}, ADV_LOSS: {}".format(
                np.mean(ae_loss), np.mean(disc_loss), np.mean(adv_loss))

    def plot(self):
        for k in xrange(self.y_dim / 10):
            plt.figure(figsize=(10, 10))
            if self.y_dim % 10 == 0 or k != (self.y_dim / 10 - 1):
                for i in xrange(100):
                    plt.subplot(10, 10, i + 1)
                    y = np.zeros(self.y_dim).astype(theano.config.floatX)
                    y[k * 10 + i / 10] = 1
                    code = np.concatenate((y, np.random.normal(0., 1., self.z_dim).astype(theano.config.floatX)))[
                           np.newaxis, :]
                    img = self.decode([code])
                    img = img.reshape((28, 28))
                    plt.imshow(img)
            else:
                nb_pic = self.y_dim % 10
                for i in xrange(nb_pic):
                    plt.subplot(nb_pic, 10, i + 1)
                    y = np.zeros(self.y_dim).astype(theano.config.floatX)
                    y[k * 10 + i / 10] = 1
                    code = np.concatenate((y, np.random.normal(0., 1., self.z_dim).astype(theano.config.floatX)))[
                           np.newaxis, :]
                    img = self.decode([code])
                    img = img.reshape((28, 28))
                    plt.imshow(img)
            plt.savefig("Cluster{}".format(k+1))


class SemiSupervisedAdversarialAutoencoder(BaseAdversarialAutoencoder):
    def __init__(self, input_shape, encoding_laye=None, decoing_layer=None, disc_layer=None, y_dim=30, z_dim=5,
                 lam=0.0001, optimizer='rmsprop', dropout=None, verbose=0, z_sample_func=None):
        super(SemiSupervisedAdversarialAutoencoder, self).__init__(input_shape, encoding_laye, decoing_layer,
                                                                   disc_layer, y_dim, z_dim, lam, optimizer,
                                                                   verbose)

        self.z_sample_func = z_sample_func if z_sample_func is not None else functools.partial(np.random.normal,
                                                                                               loc=0.,
                                                                                               scale=1.0)
        self._build_generator()
        self._build_decoder()
        self._build_discriminator()

    def plot_model(self):
        from keras.utils.visualize_util import plot
        plot(self.generator_y, to_file='generator_y.png')
        plot(self.generator_z, to_file='generator_z.png')
        plot(self.autoencoder, to_file='autoencoder.png')
        plot(self.discriminator_y, to_file='discriminator_y.png')
        plot(self.discriminator_z, to_file='discriminator_z.png')
        plot(self.discriminator_yx, to_file='discriminator_yx.png')
        plot(self.discriminator_zx, to_file='discriminator_zx.png')

    def _build_generator(self):
        super(SemiSupervisedAdversarialAutoencoder, self)._build_generator()
        self.generator_y = Model(self.x_in, self.y)
        self.generator_y.compile(self.optimizer, 'categorical_crossentropy')
        self.generator_z = Model(self.x_in, self.z)
        if 1 == self.verbose:
            print "Generator_y Summary"
            self.generator_y.summary()
            print "Generator_z Summary"
            self.generator_z.summary()

    def _build_discriminator(self):
        self.y_input = Input((self.y_dim,))
        self.h_y = self.y_input
        for i, neurons in enumerate(self.disc_layer):
            self.h_y = Dense(neurons, activation='relu', W_regularizer=l2(self.lam))(self.h_y)
        self.d_out_y = Dense(1, activation='sigmoid', W_regularizer=l2(self.lam))(self.h_y)
        self.discriminator_y = Model(self.y_input, self.d_out_y)
        self.discriminator_y.compile(self.optimizer, 'binary_crossentropy')
        if 1 == self.verbose:
            print "Discriminator_y Summary"
            print self.discriminator_y.summary()
        for l in self.discriminator_y.layers:
            l.trainable = False
        self.x_hat = Input((self.input_shape,))
        self.h_y_hat = self.discriminator_y(self.generator_y(self.x_hat))
        self.discriminator_yx = Model(self.x_hat, self.h_y_hat)
        self.discriminator_yx.compile(self.optimizer, 'binary_crossentropy')
        if 1 == self.verbose:
            print "Discriminator_yx Summary"
            self.discriminator_yx.summary()

        self.z_input = Input((self.z_dim,))
        self.h_z = self.z_input
        for i, neurons in enumerate(self.disc_layer):
            self.h_z = Dense(neurons, activation='relu', W_regularizer=l2(self.lam))(self.h_z)
        self.d_out_z = Dense(1, activation='sigmoid', W_regularizer=l2(self.lam))(self.h_z)
        self.discriminator_z = Model(self.z_input, self.d_out_z)
        self.discriminator_z.compile(self.optimizer, 'binary_crossentropy')
        if 1 == self.verbose:
            print "Discriminator_z Summary"
            self.discriminator_z.summary()
        for l in self.discriminator_z.layers:
            l.trainable = False
        self.h_z_hat = self.discriminator_z(self.generator_z(self.x_hat))
        self.discriminator_zx = Model(self.x_hat, self.h_z_hat)
        self.discriminator_zx.compile(self.optimizer, 'binary_crossentropy')
        if 1 == self.verbose:
            print "Discriminator_zx Summary"
            self.discriminator_zx.summary()

    def fit(self, X, X_labeled, y, batch_size=100, nb_epoch=10):
        y = to_categorical(y, self.y_dim)
        ae_loss = np.zeros(X.shape[0] / batch_size)
        disc_loss = np.zeros(X.shape[0] / batch_size)
        adv_loss = np.zeros(X.shape[0] / batch_size)
        for epoch in xrange(nb_epoch):
            np.random.shuffle(X)
            np.random.shuffle(X_labeled)
            for batch in xrange(X.shape[0] / batch_size):
                X_batch = X[batch * batch_size:(batch + 1) * batch_size, :]
                y_generated = to_categorical(np.random.randint(0, self.y_dim, size=X_batch.shape[0]), self.y_dim)
                z_generated = self.z_sample_func(size=(X_batch.shape[0], self.z_dim)).astype(theano.config.floatX)
                ae_loss[batch] = self.autoencoder.train_on_batch(X_batch, X_batch)
                disc_y_x = np.concatenate((self.generator_y.predict(X_batch), y_generated))
                disc_z_x = np.concatenate((self.generator_z.predict(X_batch), z_generated))
                disc_labels = np.concatenate((np.ones(X_batch.shape[0]), np.zeros(X_batch.shape[0])))
                disc_loss[batch] = self.discriminator_y.train_on_batch(disc_y_x, disc_labels) \
                                   + self.discriminator_z.train_on_batch(disc_z_x, disc_labels)
                adv_loss[batch] = self.discriminator_yx.train_on_batch(X_batch, np.zeros(X_batch.shape[0])) + \
                                  self.discriminator_zx.train_on_batch(X_batch, np.zeros(X_batch.shape[0]))
            classification_loss = self.generator_y.fit(X_labeled, y, verbose=self.verbose).history['loss']
            print "AE LOSS: {}, DISC LOSS: {}, ADV_LOSS: {}, SEMI_CLASS_LOSS: {}".format(
                np.mean(ae_loss), np.mean(disc_loss), np.mean(adv_loss), np.mean(classification_loss))

    def plot(self):
        plt.figure(figsize=(10, 10))
        for i in xrange(100):
            plt.subplot(10, 10, i + 1)
            y = np.zeros(self.y_dim).astype(theano.config.floatX)
            y[i / 10] = 1
            code = np.concatenate((y, np.random.normal(0., 1., self.z_dim).astype(theano.config.floatX)))[
                   np.newaxis, :]
            img = self.decode([code])
            img = img.reshape((28, 28))
            plt.imshow(img)
        plt.savefig("SemiSupervisedViz.png")


if __name__ == '__main__':
    from keras.datasets import mnist
    from sklearn.model_selection import train_test_split

    plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
    plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype(theano.config.floatX)
    X_train = X_train.reshape((X_train.shape[0], 784))
    X_train /= 256.
    X_train_unlabeled, X_train_labeled, y_train_unlabeled, y_train_labeled = train_test_split(X_train, y_train, test_size=0.2)
    X_test = X_test.astype(theano.config.floatX)
    X_test = X_test.reshape((X_test.shape[0], 784)).astype(theano.config.floatX)
    X_test /= 256.
    advae = SemiSupervisedAdversarialAutoencoder(784, [1000, 1000], [1000, 1000], [1000, 1000], y_dim=10, verbose=0)
    advae.fit(X_train_unlabeled, X_train_labeled, y_train_labeled, nb_epoch=5000)
    advae.plot()
    clusters = UnsupervisedCluster(784, [3000, 3000], [3000, 3000], [3000, 3000], 30, verbose=0)
    clusters.fit(X_train, nb_epoch=1500)
    clusters.plot()
