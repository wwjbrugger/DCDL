import random as random

import numpy as np
import data.mnist_fashion as fashion
import data.mnist_dataset as numbers
import data.cifar_dataset as cifar
import helper_methods as help_dither
import matplotlib.pyplot as plt

def balance_data_set(data_list, label_list, number_class_to_predict, seed = None):
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
    # how many example we need from a label which is part of 'all' other classes
    # update possibility (was not changed to be consistent with existing experiment results):
    #   use following two lines
    #   num_elements_one_class = int(label_list[:, one_class_against_all].sum())
    #   num_elements_minority = int(num_elements_one_class/ (number_classes-1))
    #   code bellow assumes all label are represented with the same number of  elements this has not to be true
    #   (splitting dataset randomly in train and val data )
    num_elements_minority = int(label_list.shape[0] / number_classes/ (number_classes-1))

    for i in range(number_classes):
        # get all indices of data which belong to label i
        indices = [j for j, one_hot_label in enumerate(label_list) if
                          one_hot_label[i] == 1]

        if i == number_class_to_predict:
            # add all examples which belong to label 'one'
            balanced_dataset.append(data_list[indices])
            balanced_label.append(label_list[indices])
        else:
            # sampel subset of label i of size num_elements_minority
            sub_sample_indices = random.sample(indices, num_elements_minority)
            balanced_dataset.append(data_list[sub_sample_indices])
            balanced_label.append(label_list[sub_sample_indices])

    balanced_dataset = np.concatenate(balanced_dataset, axis=0)
    balanced_label = np.concatenate(balanced_label, axis=0)
    # permute data so not all label are at one position of the data array
    idx = np.random.permutation(len(balanced_label))
    x, y = balanced_dataset[idx], balanced_label[idx]

    return x, y





def prepare_dataset(size_train_nn, size_valid_nn, dithering_used, one_against_all, data_set_to_use=None, seed = None):
    print("Dataset processing", flush=True)
    # load dataset
    if data_set_to_use in 'fashion':
        dataset = fashion.data(dither_method=dithering_used)
    if data_set_to_use in 'mnist':
        dataset = numbers.data(dither_method=dithering_used)
    if data_set_to_use in 'cifar':
        dataset = cifar.data(dither_method=dithering_used)

    # get data
    dataset.get_iterator()
    train_nn, label_train_nn = dataset.get_chunk(size_train_nn)
    val, label_val = dataset.get_chunk(size_valid_nn)
    test, label_test = dataset.get_test(dither_method=dithering_used)


    if data_set_to_use in 'mnist':
        class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    elif data_set_to_use in 'fashion':
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif data_set_to_use in 'cifar':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    help_dither.visualize_pic(train_nn, label_train_nn, class_names,
                    "Train data dither method: {}".format(dithering_used), plt.cm.Greys)

    train_nn, label_train_nn = balance_data_set(data_list=train_nn, label_list=label_train_nn,
                                                number_class_to_predict=one_against_all, seed = seed)
    val, label_val = balance_data_set(data_list=val, label_list=label_val,
                                      number_class_to_predict=one_against_all, seed = seed)
    test, label_test = balance_data_set(data_list=test, label_list=label_test,
                                        number_class_to_predict=one_against_all, seed = seed)

    # change label to one against all label
    label_train_nn = help_dither.one_class_against_all(label_list=label_train_nn, one_against_all=one_against_all)
    label_val = help_dither.one_class_against_all(label_list=label_val, one_against_all=one_against_all)
    label_test = help_dither.one_class_against_all(label_list=label_test, one_against_all=one_against_all)


    return train_nn, label_train_nn, val, label_val, test, label_test


def train_model(network, dithering_used, one_against_all, data_set_to_use, path_to_use,
                convert_to_grey, dither_frame, size_train_nn, size_valid_nn, seed):

    print("Training", flush=True)


    train_nn, label_train_nn, val, label_val, test, label_test = prepare_dataset(size_train_nn=size_train_nn,
                                                                                 size_valid_nn=size_valid_nn,
                                                                                 dithering_used=dithering_used,
                                                                                 one_against_all=one_against_all,
                                                                                 data_set_to_use=data_set_to_use,
                                                                                 seed = seed
                                                                                 )

    print('\n\n used data sets are saved')

    np.save(path_to_use['train_data'], train_nn)
    np.save(path_to_use['train_label'], label_train_nn)
    np.save(path_to_use['val_data'], val)
    np.save(path_to_use['val_label'], label_val)
    np.save(path_to_use['test_data'], test)
    np.save(path_to_use['test_label'], label_test)



    print("Start Training")
    network.training(train_nn, label_train_nn, val, label_val, path_to_use)

    # evaluate trained net on the training data set and save results
    print("\n Start evaluate with train set ")
    evaluation_result = network.evaluate(train_nn, label_train_nn)

    dither_frame.at[0, '{}_Train'.format(dithering_used)] = evaluation_result

    # evaluate trained net on validation set
    print("\n Start evaluate with validation set ")
    network.evaluate(val, label_val)

    # evaluate trained net on test set and save result
    print("\n Start evaluate with test set ")
    evaluation_result = network.evaluate(test, label_test)

    dither_frame.at[0, '{}_Test'.format(dithering_used)] = evaluation_result



    print('end')

def train_further(network):
    network.training

