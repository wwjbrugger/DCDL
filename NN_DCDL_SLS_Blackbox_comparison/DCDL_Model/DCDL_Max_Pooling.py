
from skimage.measure import block_reduce
import numpy as np


class Max_Pooling:
    # class to approximate a max_pooling layer of the neural net
    def __init__(self, properties):
        self.properties = properties


    def preprocess(self, data, label):
        # calculate max pooling operation
        data_after_max_pooling = block_reduce(
            data,
            block_size=self.properties['block_size'],
            func=np.max)


        return data_after_max_pooling, label



    def train(self, train_data, train_label, validation_data, validation_label):
        # no training needed
        pass


    def prediction(self, data, original_label):
        # calculate max pooling operation
        data_after_max_pooling, _ = self.preprocess(
            data=data,
            label=original_label)
        # acc is calculated by DCDL class
        acc = None

        return data_after_max_pooling, acc
