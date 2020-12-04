import numpy as np
import tensorflow as tf
import os
import compare_dither.dithering_diffusion as dithering_diffusion
import random


def num_variables():
    return 28 * 28

class data():
    def __init__(self, dither_method = False):
        """

        @param dither_method:
        """
        if dither_method:
            dirname = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(dirname, os.path.join('data/Fashion_dither/', dither_method))
            if not os.path.exists(path):
                os.mkdir(path)
            if os.path.exists(os.path.join(path, 'train.npy')) and  os.path.exists(os.path.join(path, 'label_train.npy')):
                train= np.load(os.path.join(path, 'train.npy'))
                label_train = np.load(os.path.join(path, 'label_train.npy'))
            else:
                (train, label_train), _ = tf.keras.datasets.fashion_mnist.load_data()
                train = train.reshape((train.shape + (1,)))
                train = dithering_diffusion.error_diffusion_dithering(train, dither_method)
                # set black pixel to the value -1
                train= np.where(train, 1, -1)
                path_to_save = os.path.join(path, 'train.npy')
                np.save(path_to_save,train)

                np.save(os.path.join(path, 'label_train.npy'), label_train)
        else:
            (train, label_train), _ = tf.keras.datasets.fashion_mnist.load_data()

        targets = np.array([label_train]).reshape(-1)
        self.label_train = np.eye(classes)[targets]

        first_half = train.astype(np.float32)
        self.train = first_half / 255

    def get_iterator(self):
        p = np.random.permutation(self.train.shape[0])
        self.label_train, self.train = self.label_train[p], self.train[p]
        self.iter = 0

    def get_chunk(self, chunk_size):
        if self.iter + chunk_size > end:
            return None
        t_ret, l_ret = self.train[self.iter: self.iter +
                                  chunk_size], self.label_train[self.iter: self.iter+chunk_size]
        self.iter += chunk_size
        return np.array(t_ret), np.array(l_ret)

    def get_test(self, dither_method = False):

        if dither_method:
            dirname = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(dirname, os.path.join('data/Fashion_dither/', dither_method))
            if not os.path.exists(path):
                os.mkdir(path)
            if os.path.exists(os.path.join(path, 'test.npy')) and os.path.exists(os.path.join(path, 'label_test.npy')):
                test = np.load(os.path.join(path, 'test.npy'))
                label_test = np.load(os.path.join(path, 'label_test.npy'))
            else:

                _, (test, label_test) = tf.keras.datasets.fashion_mnist.load_data()
                test = test.reshape((test.shape + (1,)))
                test = dithering_diffusion.error_diffusion_dithering(test, dither_method)
                # set black pixel to the value -1
                test = np.where(test, 1, -1)
                path_to_save = os.path.join(path, 'test.npy')
                np.save(path_to_save, test)

                np.save(os.path.join(path, 'label_test.npy'), label_test)
        else:
            _, (test, label_test) = tf.keras.datasets.fashion_mnist.load_data() # Returns: Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).


        targets = np.array([label_test]).reshape(-1)
        label_test = np.eye(classes)[targets]

        first_half = test.astype(np.float32)/255
        return first_half, label_test

    def get_name(self):
        return 'Fashion'


classes = 10
end = 60000

