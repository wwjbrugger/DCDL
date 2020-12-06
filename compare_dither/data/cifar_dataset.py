import numpy as np
import tensorflow as tf
import random
import os
import compare_dither.dithering_diffusion as dithering_diffusion

# Update possibility (was not changed to be consistent with existing experiment results):
#     remove method not used
def num_variables():
    return 32 * 32 * 150


class data():

    def __init__(self, dither_method=False):
        """

        @param dither_method:
        """
        if dither_method:
            # if data should be dithered
            dirname = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(dirname, os.path.join('data/Cifar_dither/', dither_method))
            if not os.path.exists(path):
                # if path to data not exist, create it
                os.mkdir(path)
            if os.path.exists(os.path.join(path, 'train.npy')) and os.path.exists(
                    os.path.join(path, 'label_train.npy')):
                # if data has already been dithered with this method, load it
                train = np.load(os.path.join(path, 'train.npy'))
                label_train = np.load(os.path.join(path, 'label_train.npy'))
            else:
                # data has not been dithered yet with this method
                # data format train: [num_samples, width, height, colour_channel] pixel range is [0 to 255]
                # data format label: [[label], [label], ... ]
                (train, label_train), _ = tf.keras.datasets.cifar10.load_data()

                # scale pixel range to [ 0 to 1]
                train = train/255
                # dither data. can take a while
                train = dithering_diffusion.error_diffusion_dithering(train, dither_method)
                # save dithered data
                path_to_save = os.path.join(path, 'train.npy')
                np.save(path_to_save, train)

                np.save(os.path.join(path, 'label_train.npy'), label_train)
        else:
            # if data should not be dithered
            (train, label_train), _ = tf.keras.datasets.cifar10.load_data()
            # scale pixel range to [ 0 to 1]
            train = train / 255

        # flatten label
        targets = np.array([label_train]).reshape(-1)
        # cast to one hot vectors
        self.label_train = np.eye(classes)[targets]

        first_half = train.astype(np.float32)
        self.train = first_half


    def get_iterator(self):
        # permutes data and label
        # set iter to 0 if n data are requested return next n and set iter to iter = iter + n
        p = np.random.permutation(self.train.shape[0])
        self.label_train, self.train = self.label_train[p], self.train[p]
        self.iter = 0

    def get_chunk(self, chunk_size):
        if self.iter + chunk_size > end:
            # ask for more data then are available
            # Update possibility (was not changed to be consistent with existing experiment results):
            # raise error
            return None
        # get chunck_size many data
        t_ret, l_ret = self.train[self.iter: self.iter +
                                  chunk_size], self.label_train[self.iter: self.iter+chunk_size]
        # update position of  iter
        self.iter += chunk_size
        return np.array(t_ret), np.array(l_ret)

    def get_test(self, dither_method=False):


        if dither_method:
            # if data should be dithered
            dirname = os.path.dirname(os.path.dirname(__file__))
            path = os.path.join(dirname, os.path.join('data/Cifar_dither/', dither_method))
            if not os.path.exists(path):
                # if path to data not exist, create it
                os.mkdir(path)
            if os.path.exists(os.path.join(path, 'test.npy')) and os.path.exists(os.path.join(path, 'label_test.npy')):
                # if data has already been dithered with this method, load it
                test = np.load(os.path.join(path, 'test.npy'))
                label_test = np.load(os.path.join(path, 'label_test.npy'))
            else:
                # data has not been dithered yet with this method
                # data format train: [num_samples, width, height, colour_channel] pixel range is [0 to 255]
                # data format label: [[label], [label], ... ]
                _, (test, label_test) = tf.keras.datasets.cifar10.load_data()
                # scale pixel range to [ 0 to 1]
                test = test/255
                # dither data. can take a while
                test = dithering_diffusion.error_diffusion_dithering(test, dither_method)

                # save dithered data
                path_to_save = os.path.join(path, 'test.npy')
                np.save(path_to_save, test)
                np.save(os.path.join(path, 'label_test.npy'), label_test)
        else:
            # if data should not be dithered
            _, (test, label_test) = tf.keras.datasets.cifar10.load_data()
            # scale pixel range to [ 0 to 1]
            test = test/255

        #  flatten label
        targets = np.array([label_test]).reshape(-1)
        # cast to one hot vectors
        label_test = np.eye(classes)[targets]

        first_half = test.astype(np.float32)
        return first_half, label_test


    def get_name(self):
        return 'Cifar'

 # Update possibility (was not changed to be consistent with existing experiment results):
            # set as class property
classes = 10
end = 50000
