import numpy as np
import tensorflow as tf
import random




class data():
    def __init__(self):
        # data format train: [num_samples, width, height] pixel range is [0 to 255]
        #                     no colour channel !
        # data format label: [[label], [label], ... ]
        (train, label_train), _ = tf.keras.datasets.mnist.load_data()
        # how many classes are in data set
        self.classes = 10
        # size of original dataset
        self.end = 60000
        # flatten label
        targets = np.array([label_train]).reshape(-1)
        # cast to one hot vectors
        self.label_train = np.eye(classes)[targets]

        first_half = train.astype(np.float32)
        # scale pixel range to [ 0 to 1]
        self.train = first_half / 255

    def get_iterator(self):
        # permutes data and label
        # set iter to 0 if n data are requested return next n and set iter to iter = iter + n
        p = np.random.permutation(self.train.shape[0])
        self.label_train, self.train = self.label_train[p], self.train[p]
        self.iter = 0

    def get_chunk(self, chunk_size):
        if self.iter + chunk_size > self.end:
            # ask for more data then are available
            # raise error
            raise IndexError('You requested more data as there are in the dataset')

        # get chunck_size many data
        t_ret, l_ret = self.train[self.iter: self.iter +
                                  chunk_size], self.label_train[self.iter: self.iter+chunk_size]
        # update position of iter
        self.iter += chunk_size
        return np.array(t_ret), np.array(l_ret)

    def get_test(self):
        # data format train: [num_samples, width, height] pixel range is [0 to 255]
        #                     no colour channel !
        # data format label: [[label], [label], ... ]
        _, (test, label_test) = tf.keras.datasets.mnist.load_data() # Returns: Tuple of Numpy arrays: (x_train, y_train), (x_test, y_test).
        # flatten label
        targets = np.array([label_test]).reshape(-1)
        # cast to one hot vectors
        label_test = np.eye(self.classes)[targets]

        # scale pixel range to [ 0 to 1]
        first_half = test.astype(np.float32)/255
        return first_half, label_test




