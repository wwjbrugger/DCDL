import numpy as np
import tensorflow as tf
import random


def num_variables():
    return 32 * 32 * 150


class data():
    def __init__(self):
        (train, label_train), _ = tf.keras.datasets.cifar10.load_data()
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

    def get_test(self):
        _, (test, label_test) = tf.keras.datasets.cifar10.load_data()
        targets = np.array([label_test]).reshape(-1)
        label_test = np.eye(classes)[targets]

        first_half = test.astype(np.float32)/255
        return first_half, label_test


classes = 10
end = 50000
