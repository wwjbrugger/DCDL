import random as random

import numpy as np
import comparison_DCDL_vs_SLS.data.fashion.mnist_fashion as fashion
import comparison_DCDL_vs_SLS.data.numbers.mnist_dataset as numbers
import comparison_DCDL_vs_SLS.data.cifar.cifar_dataset as cifar
import comparison_DCDL_vs_SLS.dithering as dith
import helper_methods as helper_methods
import matplotlib.pyplot as plt



# def balance_data_set(data, label, percent_of_major_label_to_keep):
#     old balance method
#     """
#     :param data:  dithered data
#     :param label: label in one hot encoding. [1,0] data is part of the 'one' label.   [0,1] data is part of the 'all' label.
#     :param percent_of_major_label_to_keep:
#     :return:
#     """
#     unique, counts = np.unique(label[:, 0], return_counts=True)
#     print('size_of_set before balancing {} with {}'.format(data.shape[0], dict(zip(unique, counts))))
#     # keeps data if label is part of the one label and
#     # by a chance of percent_of_major_label_to_keep if it's a rest class
#     index_to_keep = [i for i, one_hot_label in enumerate(label) if
#                      one_hot_label[0] == 1 or random.random() < percent_of_major_label_to_keep]
#     # get balanced data and label
#     data = data[index_to_keep]
#     label = label[index_to_keep]
#
#     unique, counts = np.unique(label[:, 0], return_counts=True)
#     print('size_of_set {} with {}'.format(data.shape[0], dict(zip(unique, counts))))
#     return data, label

def balance_data_set(data_list, label_list, one_class_against_all, seed = None):
    """
    Balances data for one against all tests.
    Half of the returned data is from label 'one' class
    The other half consists of equal parts of 'all' other classes.
    @param data_list:
    @param label_list:
    @param number_class_to_predict:
    @param seed:
    @return:
    """
    np.random.seed(seed)
    number_classes = label_list.shape[1]
    balanced_dataset = []
    balanced_label = []
    # how many example we have in the set of the 'one' class  from a label which is part of 'all' other classes
    num_elements_one_class = int(label_list[:, one_class_against_all].sum())
    num_elements_minority = int(num_elements_one_class/ (number_classes-1))

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


def prepare_dataset(size_train_nn, size_valid_nn, dithering_used, one_against_all,
                    number_class_to_predict, data_set_to_use, convert_to_grey, values_max_1):
    # preprocess datasets
    #   - load
    #   - dither
    #   - get label for one_class_against_all testing
    # set which data set should be used
    # balance dataset
    print("Dataset processing", flush=True)
    if data_set_to_use in 'fashion':
        dataset = fashion.data()
    if data_set_to_use in 'numbers':
        dataset = numbers.data()
    if data_set_to_use in 'cifar':
        dataset = cifar.data()
    dataset.get_iterator()

    # The original train dataset which is loaded from tensorflow is permuted.
    # In a sequential manner the first size_train_nn are returned
    # if validation data are requested the next size_valid_nn data are returned
    train_nn, label_train_nn = dataset.get_chunk(chunk_size=size_train_nn)
    val, label_val = dataset.get_chunk(chunk_size=size_valid_nn)
    # The original test dataset which is loaded from tensorflow is permuted and then returned.
    test, label_test = dataset.get_test()
    if train_nn.ndim == 3:
        # Add an extra colour channel to data which are only in shape [number pictures, width, height]
        train_nn = train_nn.reshape((train_nn.shape + (1,)))
        val = val.reshape((val.shape + (1,)))
        test = test.reshape((test.shape + (1,)))

    # define caption for showing dataset
    if data_set_to_use in 'numbers':
        class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    elif data_set_to_use in 'fashion':
        class_names =  ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif data_set_to_use in 'cifar':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # show data before preprocessing
    dith.visualize_pic(pic_array=train_nn,
                       label_array=label_train_nn,
                       class_names=class_names,
                       title=" pic how they are in dataset",
                       colormap=plt.cm.Greys)

    # balance dataset otherwise we would have 10 % one Label data and 90 %  rest data
    train_nn, label_train_nn = balance_data_set(data_list=train_nn,
                                                label_list=label_train_nn,
                                                one_class_against_all=one_against_all)
    val, label_val = balance_data_set(data_list=val,
                                      label_list=label_val,
                                      one_class_against_all=one_against_all)
    test, label_test = balance_data_set(data_list=test,
                                        label_list=label_test,
                                        one_class_against_all=one_against_all)

    # convert colour picture in greyscale pictures. Is not used in experiment
    if convert_to_grey:
        train_nn = helper_methods.convert_to_grey(pic_array = train_nn)
        val = helper_methods.convert_to_grey(pic_array = val)
        test = helper_methods.convert_to_grey(pic_array= test)
        dith.visualize_pic(pic_array=train_nn,
                           label_array=label_train_nn,
                           class_names=class_names,
                           title=" pic in gray ",
                           colormap=plt.cm.Greys)

    # dither data with 'floyd-steinberg' algorithm in Pillow library
    if dithering_used:
        train_nn = dith.dither_pic(pic_array = train_nn,
                                   values_max_1 = values_max_1)
        val = dith.dither_pic(pic_array = val,
                              values_max_1= values_max_1)
        test = dith.dither_pic(pic_array=test,
                               values_max_1= values_max_1)

        dith.visualize_pic(pic_array=train_nn,
                           label_array=label_train_nn,
                           class_names=class_names,
                           title = " pic after dithering",
                           colormap= plt.cm.Greys)

    # convert for one-against-all testing one-hot label with 10 classes in one-hot label with two classes ([one, rest])
    label_train_nn = helper_methods.one_class_against_all(array_label = label_train_nn,
                                                one_class= one_against_all,
                                                number_classes_output = number_class_to_predict)
    label_val = helper_methods.one_class_against_all(array_label =  label_val,
                                           one_class = one_against_all,
                                           number_classes_output = number_class_to_predict)
    label_test = helper_methods.one_class_against_all(array_label = label_test,
                                            one_class = one_against_all,
                                            number_classes_output=number_class_to_predict)



    # show data after preprocessing
    dith.visualize_pic(pic_array = train_nn,
                       label_array = label_train_nn,
                       class_names = class_names,
                       title = " pic how they are feed into net",
                       colormap =plt.cm.Greys)

    return train_nn, label_train_nn, val, label_val, test, label_test


def train_model(network, dithering_used, one_against_all, data_set_to_use, path_to_use, convert_to_grey,
                results, size_train_nn, size_valid_nn, values_max_1):

    print("Training", flush=True)

    # preprocess datasets
    train_nn, label_train_nn, val, label_val, test, label_test = prepare_dataset(size_train_nn = size_train_nn,
                                                                                 size_valid_nn=size_valid_nn,
                                                                                 dithering_used=dithering_used,
                                                                                 one_against_all=one_against_all,
                                                                                 number_class_to_predict=network.classes,
                                                                                 data_set_to_use=data_set_to_use,
                                                                                 convert_to_grey = convert_to_grey,
                                                                                 values_max_1 = values_max_1)

    print('\n\n used data sets are saved')
    # save preprocessed datasets (needed for later evaluation)
    np.save(file=path_to_use['train_data'], arr=train_nn)
    np.save(file=path_to_use['train_label'],arr=label_train_nn)
    np.save(file=path_to_use['val_data'], arr=val)
    np.save(file=path_to_use['val_label'], arr=label_val)
    np.save(file=path_to_use['test_data'], arr=test)
    np.save(file=path_to_use['test_label'], arr=label_test)



    print("Start Training")
    # train network
    network.training(train=train_nn,
                     label_train=label_train_nn,
                     val=val,
                     label_val=label_val,
                     path_to_use=path_to_use)

    print("\n Start evaluate with train set ")
    # save accuracy on the NN on train set in results
    results.at[1, 'Neural network'] = network.evaluate(input=train_nn, label=label_train_nn)

    print("\n Start evaluate with validation set ")
    # should be the same value as the highest during training
    network.evaluate(input=val, label=label_val)

    print("\n Start evaluate with test set ")
    # save accuracy of the NN on test set in results
    results.at[3, 'Neural network'] = network.evaluate(input=test, label=label_test)

    print('end')
