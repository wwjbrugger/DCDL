import numpy as np
import random as random

import get_data_scripts_and_dither.mnist_fashion as fashion
import get_data_scripts_and_dither.cifar_dataset as cifar
import get_data_scripts_and_dither.mnist_numbers as numbers

import get_data_scripts_and_dither.dithering as dith
import NN_DCDL_SLS_Blackbox_comparision.visualize as visualize


def get_data(size_train, size_valid, dithering_used, one_against_all, number_class_to_predict,
             data_set_to_use, convert_to_grey, values_max_1, visualize_data):
    """

    :param visualize_data: if data should be shown during preprocessing
    :param number_class_to_predict:  length of one-hot-encoded label e.g.[0,1], after one_against_all:
    :param size_train: size of train set
    :param size_valid: size of validation set
    :param dithering_used: If pictures should be dithered set values to 0 or 1
    :param one_against_all: class to test rest against
    :param data_set_to_use: which data set to use
    :param convert_to_grey: if pictures should be converted to grey before using
    :param values_max_1: are pixel of pictures in range [0 to 1]? other option [0 to 255]
    :return: train, validation and test dataset in bool format [True, False]
    """

    # --------------------------get raw data-----------------------------
    print("Dataset processing", flush=True)
    if data_set_to_use in 'fashion':
        dataset = fashion.data()
    if data_set_to_use in 'numbers':
        dataset = numbers.data()
    if data_set_to_use in 'cifar':
        dataset = cifar.data()
    dataset.get_iterator()

    # The original train dataset which is loaded from tensorflow is permuted.
    # In a sequential manner the first size_train are returned
    # if validation data are requested the next size_valid data are returned
    train, label_train = dataset.get_chunk(chunk_size=size_train)
    val, label_val = dataset.get_chunk(chunk_size=size_valid)
    # The original test dataset which is loaded from tensorflow is permuted and then returned.
    test, label_test = dataset.get_test()

    # -------------------------------add colour channel if necessarily-------------
    if train.ndim == 3:
        # Add an extra colour channel to data which are only in shape [number pictures, width, height]
        train = train.reshape((train.shape + (1,)))
        val = val.reshape((val.shape + (1,)))
        test = test.reshape((test.shape + (1,)))
    # ------------------------------ visualize raw data----------------------------
    if visualize_data:
        # define caption for showing dataset
        if data_set_to_use in 'numbers':
            class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        elif data_set_to_use in 'fashion':
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
                           'Ankle boot']
        elif data_set_to_use in 'cifar':
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        # show data before preprocessing
        visualize.visualize_pic(pic_array=train,
                                label_array=label_train,
                                class_names=class_names,
                                title=" pic how they are in dataset")

    # ----------------------------------balance data-------------------------------

    # balance dataset otherwise we would have 10 % one Label data and 90 %  rest data
    train, label_train = balance_data_set(data_list=train,
                                          label_list=label_train,
                                          one_class_against_all=one_against_all)
    val, label_val = balance_data_set(data_list=val,
                                      label_list=label_val,
                                      one_class_against_all=one_against_all)
    test, label_test = balance_data_set(data_list=test,
                                        label_list=label_test,
                                        one_class_against_all=one_against_all)
    # ---------------------------------- convert to grey--------------------------------
    # convert colour picture in greyscale pictures. Is not used in experiment
    if convert_to_grey:
        train = convert_to_grey(pic_array=train)
        val = convert_to_grey(pic_array=val)
        test = convert_to_grey(pic_array=test)
        if visualize_data:
            visualize.visualize_pic(pic_array=train,
                                    label_array=label_train,
                                    class_names=class_names,
                                    title=" pic in gray ",
                                    )
    # -----------------------------------dither data--------------------
    # dither data with 'floyd-steinberg' algorithm in Pillow library
    if dithering_used:
        train = dith.dither_pic(pic_array=train,
                                values_max_1=values_max_1)
        val = dith.dither_pic(pic_array=val,
                              values_max_1=values_max_1)
        test = dith.dither_pic(pic_array=test,
                               values_max_1=values_max_1)
        if visualize_data:
            visualize.visualize_pic(pic_array=train,
                                    label_array=label_train,
                                    class_names=class_names,
                                    title=" pic after dithering")
        # ----------------------------- cast data to boolean values-------------------

        train = transform_to_boolean(train)
        val = transform_to_boolean(val)
        test = transform_to_boolean(test)

        if visualize_data:
            # show data after preprocessing
            visualize.visualize_pic(pic_array=np.float32(train),
                                    label_array=label_train,
                                    class_names=class_names,
                                    title=" pic transformed to boolean ")

    # ----------------------------------convert to one-against-all-lable ---------
    # convert for one-against-all testing one-hot label with 10 classes in one-hot label with two classes ([one, rest])
    label_train = cast_to_one_class_against_all(array_label=label_train,
                                                one_class=one_against_all,
                                                number_classes_output=number_class_to_predict,
                                                kind_of_data='train')
    label_val = cast_to_one_class_against_all(array_label=label_val,
                                              one_class=one_against_all,
                                              number_classes_output=number_class_to_predict,
                                              kind_of_data='validation')
    label_test = cast_to_one_class_against_all(array_label=label_test,
                                               one_class=one_against_all,
                                               number_classes_output=number_class_to_predict,
                                               kind_of_data='test')
    if visualize_data:
        # show data after preprocessing
        visualize.visualize_pic(pic_array=np.float32(train),
                                label_array=label_train,
                                class_names=['main', 'rest'],
                                title=" pic how they are used", )
    data_dic = {
        'train': train,
        'label_train': label_train,
        'val': val,
        'label_val': label_val,
        'test': test,
        'label_test': label_test}

    return data_dic


def balance_data_set(data_list, label_list, one_class_against_all):
    """
    Balances data for one against all tests.
    Half of the returned data is from label 'one' class
    The other half consists of equal parts of 'all' other classes.
    # input one-hot-vector [[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.], [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.], [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.], [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.], [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
    """
    number_classes = label_list.shape[1]
    balanced_dataset = []
    balanced_label = []
    # how many example we have in the set of the 'one' class  from a label which is part of 'all' other classes
    num_elements_one_class = int(label_list[:, one_class_against_all].sum())
    num_elements_minority_exact = num_elements_one_class / (number_classes - 1)
    num_elements_minority = int(np.around(num_elements_minority_exact))

    for i in range(number_classes):
        # get all indices of data which belong to label i
        indices = [j for j, one_hot_label in enumerate(label_list) if
                   one_hot_label[i] == 1]

        if i == one_class_against_all:
            # add all examples which belong to label 'one'
            balanced_dataset.append(data_list[indices])
            balanced_label.append(label_list[indices])
        else:
            # sampel subset of label i of size num_elements_minority
            # Chooses k unique random elements from a population sequence or set.
            sub_sample_indices = random.sample(population=indices, k=num_elements_minority)
            balanced_dataset.append(data_list[sub_sample_indices])
            balanced_label.append(label_list[sub_sample_indices])
    balanced_dataset = np.concatenate(balanced_dataset, axis=0)
    balanced_label = np.concatenate(balanced_label, axis=0)
    # permute data so not all label are at one position of the data array
    idx = np.random.permutation(len(balanced_label))
    x, y = balanced_dataset[idx], balanced_label[idx]

    return x, y


def cast_to_one_class_against_all(array_label, one_class, number_classes_output, kind_of_data):
    """
    converts an array with one_hot_vector for any number of classes into a one_hot_vector,
    whether an example belongs to one class or not
    """
    shape_output = (len(array_label), number_classes_output)
    label_one_class_against_all = np.zeros(shape_output, dtype=int)
    for i, one_hot_vector in enumerate(array_label):
        if one_hot_vector.argmax() == one_class:
            label_one_class_against_all[i, 0] = 1
        else:
            label_one_class_against_all[i, -1] = 1
    num_elements_one_class = int(label_one_class_against_all[:, 0].sum())
    num_elements_rest_class = int(label_one_class_against_all[:, 1].sum())
    print('{}   number one label in set: {}     number rest label in set {} '.format(
        kind_of_data, num_elements_one_class, num_elements_rest_class))
    return label_one_class_against_all


def transform_to_boolean(array):
    # cast data from -1 to 0.
    # 0 is interpreted as False
    # 1 as True
    boolean_array = np.maximum(array, 0).astype(np.bool)
    return boolean_array

def transform_boolean_to_minus_one_and_one(array):
    # cast True to 1 and False to -1
    array_minus_one_and_one = np.where(array,1,-1)
    return array_minus_one_and_one



def convert_to_grey(pic_array):
    """ convert rgb pictures in grey scale pictures """
    pictures_grey = np.empty((pic_array.shape[0], pic_array.shape[1], pic_array.shape[2], 1))
    for i, pic in enumerate(pic_array):
        pictures_grey[i, :, :, 0] = np.dot(pic[:, :, :3], [0.2989, 0.5870, 0.1140])
    return pictures_grey


def transform_label_to_one_hot(label, using_argmin_label):
    if np.ndim(label) != 1:
        raise ValueError('Label array have to be one dimensional \n'
                         'the input dimension is {}'.format(label.shape()))
    if using_argmin_label:
        # argmin function was used to cast one hot label to one number
        one_hot_label = np.array([[1,0] if l else [0,1] for l in label])
    else:
        # argmax function was used to cast one hot label to to one number
        one_hot_label = np.array([[0, 1] if l else [1, 0] for l in label])
    return one_hot_label