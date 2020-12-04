import numpy as np
import tensorflow as tf
import random
import os
import dithering_diffusion

def num_variables():
    return 32 * 32 * 150


class data():

    def __init__(self, dither_method=False):
        """

        @param dither_method:
        """
        if dither_method:
            dirname = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(dirname, os.path.join('data/Cifar_dither/', dither_method))
            if not os.path.exists(path):
                os.mkdir(path)
            if os.path.exists(os.path.join(path, 'train.npy')) and os.path.exists(
                    os.path.join(path, 'label_train.npy')):
                train = np.load(os.path.join(path, 'train.npy'))
                label_train = np.load(os.path.join(path, 'label_train.npy'))
            else:
                (train, label_train), _ = tf.keras.datasets.cifar10.load_data()


                train = train/255
                train = dithering_diffusion.error_diffusion_dithering(train, dither_method)
                path_to_save = os.path.join(path, 'train.npy')
                np.save(path_to_save, train)

                np.save(os.path.join(path, 'label_train.npy'), label_train)
        else:
            (train, label_train), _ = tf.keras.datasets.cifar10.load_data()

            train = train / 255

        targets = np.array([label_train]).reshape(-1)
        self.label_train = np.eye(classes)[targets]

        first_half = train.astype(np.float32)
        self.train = first_half


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

    def get_test(self, dither_method=False):


        if dither_method:
            dirname = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(dirname, os.path.join('data/Cifar_dither/', dither_method))
            if not os.path.exists(path):
                os.mkdir(path)
            if os.path.exists(os.path.join(path, 'test.npy')) and os.path.exists(os.path.join(path, 'label_test.npy')):
                test = np.load(os.path.join(path, 'test.npy'))

                label_test = np.load(os.path.join(path, 'label_test.npy'))
            else:

                _, (test, label_test) = tf.keras.datasets.cifar10.load_data()


                test = test/255

                test = dithering_diffusion.error_diffusion_dithering(test, dither_method)
                path_to_save = os.path.join(path, 'test.npy')
                np.save(path_to_save, test)

                np.save(os.path.join(path, 'label_test.npy'), label_test)
        else:
            _, (test, label_test) = tf.keras.datasets.cifar10.load_data()
            test = test/255


        targets = np.array([label_test]).reshape(-1)
        label_test = np.eye(classes)[targets]

        first_half = test.astype(np.float32)
        return first_half, label_test


    def get_name(self):
        return 'Cifar'

classes = 10
end = 50000
