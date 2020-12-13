
from skimage.measure import block_reduce
import numpy as np


class Max_Pooling:
    def __init__(self, properties):
        self.properties = properties


    def preprocess(self, data, label):
        data_after_max_pooling = block_reduce(
            data,
            block_size=self.properties['block_size'],
            func=np.max)


        return data_after_max_pooling, label



    def train(self, train_data, train_label, validation_data, validation_label):
        pass


    def prediction(self, data, original_label):
        data_after_max_pooling, _ = self.preprocess(
            data=data,
            label=original_label)
        acc = None

        return data_after_max_pooling, acc
