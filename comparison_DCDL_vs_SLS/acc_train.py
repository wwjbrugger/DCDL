import random as random

import numpy as np
import comparison_DCDL_vs_SLS.data.fashion.mnist_fashion as fashion
import comparison_DCDL_vs_SLS.data.numbers.mnist_dataset as numbers
import comparison_DCDL_vs_SLS.data.cifar.cifar_dataset as cifar
import comparison_DCDL_vs_SLS.dithering as dith
import helper_methods as help
import matplotlib.pyplot as plt



def balance_data_set(data, label, percent_of_major_label_to_keep):
    """
    :param data:  dithered data
    :param label: label in one hot encoding. [1,0] data is part of the 'one' label.   [0,1] data is part of the 'all' label.
    :param percent_of_major_label_to_keep:
    :return:
    """
    unique, counts = np.unique(label[:, 0], return_counts=True)
    print('size_of_set before balancing {} with {}'.format(data.shape[0], dict(zip(unique, counts))))
    # keeps data if label is part of the one label and
    # by a chance of percent_of_major_label_to_keep if it's a rest class
    index_to_keep = [i for i, one_hot_label in enumerate(label) if
                     one_hot_label[0] == 1 or random.random() < percent_of_major_label_to_keep]
    # get balanced data and label
    data = data[index_to_keep]
    label = label[index_to_keep]

    unique, counts = np.unique(label[:, 0], return_counts=True)
    print('size_of_set {} with {}'.format(data.shape[0], dict(zip(unique, counts))))
    return data, label

# Update possibility (was not changed to be consistent with existing experiment results):
#    use balance method below. This one makes sure:
#          Half of the returned data is from label 'one' class
#          The other half consists of equal parts of 'all' other classes.
# def balance_data_set(data_list, label_list, number_class_to_predict, seed = None):
#     """
#     Balances data for one against all tests.
#     Half of the returned data is from label 'one' class
#     The other half consists of equal parts of 'all' other classes.
#     @param data_list:
#     @param label_list:
#     @param number_class_to_predict:
#     @param seed:
#     @return:
#     """
#     np.random.seed(seed)
#     number_classes = label_list.shape[1]
#     balanced_dataset = []
#     balanced_label = []
#     # how many example we need from a label which is part of 'all' other classes
#     num_elements_minority = int(label_list.shape[0] / number_classes/ (number_classes-1))
#
#     for i in range(number_classes):
#         # get all indices of data which belong to label i
#         indices = [j for j, one_hot_label in enumerate(label_list) if
#                           one_hot_label[i] == 1]
#
#         if i == number_class_to_predict:
#             # add all examples which belong to label 'one'
#             balanced_dataset.append(data_list[indices])
#             balanced_label.append(label_list[indices])
#         else:
#             # sampel subset of label i of size num_elements_minority
#             sub_sample_indices = random.sample(indices, num_elements_minority)
#             balanced_dataset.append(data_list[sub_sample_indices])
#             balanced_label.append(label_list[sub_sample_indices])
#     balanced_dataset = np.concatenate(balanced_dataset, axis=0)
#     balanced_label = np.concatenate(balanced_label, axis=0)
#     # permute data so not all label are at one position of the data array
#     idx = np.random.permutation(len(balanced_label))
#     x, y = balanced_dataset[idx], balanced_label[idx]
#
#     return x, y


def prepare_dataset(size_train_nn, size_valid_nn, dithering_used, one_against_all,
                    percent_of_major_label_to_keep=None, number_class_to_predict=None, data_set_to_use=None, convert_to_grey = None):
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
    train_nn, label_train_nn = dataset.get_chunk(size_train_nn)
    val, label_val = dataset.get_chunk(size_valid_nn)
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
        # Update possibility (was not changed to be consistent with existing experiment results):
        # class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    elif data_set_to_use in 'cifar':
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    # show data before preprocessing
    dith.visualize_pic(train_nn, label_train_nn, class_names,
                       " pic how they are in dataset", plt.cm.Greys)

    # convert colour picture in greyscale pictures. Is not used in experiment
    if convert_to_grey:
        train_nn = help.convert_to_grey(train_nn)
        val = help.convert_to_grey(val)
        test = help.convert_to_grey(test)
        dith.visualize_pic(train_nn, label_train_nn, class_names,
                           " pic in gray ", plt.cm.Greys)

    # dither data with 'floyd-steinberg' algorithm in Pillow library
    if dithering_used:
        train_nn = dith.dither_pic(train_nn)
        val = dith.dither_pic(val)
        test = dith.dither_pic(test)
        dith.visualize_pic(train_nn, label_train_nn, class_names,
                           " pic after dithering", plt.cm.Greys)

    # convert for one-against-all testing one-hot label with 10 classes in one-hot label with two classes ([one, rest])
    label_train_nn = help.one_class_against_all(label_train_nn, one_against_all,
                                                number_classes_output=number_class_to_predict)
    label_val = help.one_class_against_all(label_val, one_against_all, number_classes_output=number_class_to_predict)
    label_test = help.one_class_against_all(label_test, one_against_all, number_classes_output=number_class_to_predict)

    # balance dataset otherwise we would have 10 % one Label data and 90 %  rest data
    train_nn, label_train_nn = balance_data_set(train_nn, label_train_nn, percent_of_major_label_to_keep)
    val, label_val = balance_data_set(val, label_val, percent_of_major_label_to_keep)
    test, label_test = balance_data_set(test, label_test, percent_of_major_label_to_keep)

    # show data after preprocessing
    dith.visualize_pic(train_nn, label_train_nn, class_names,
                       " pic how they are feed into net", plt.cm.Greys)

    return train_nn, label_train_nn, val, label_val, test, label_test


def train_model(network, dithering_used, one_against_all, data_set_to_use, path_to_use, convert_to_grey, results):
    if data_set_to_use in 'cifar':
        size_train_nn = 45000
    else:
        size_train_nn = 55000
    size_valid_nn = 5000
    # percentage with which data of the rest class are getting into used data sets
    # Update possibility (was not changed to be consistent with existing experiment results):
    # 0.1 is not optimal since we use the 'one' label completely we would have to use 100/9 = 11.111111 period
    # as percentage to get a 50/ 50 data set
    percent_of_major_label_to_keep = 0.1

    print("Training", flush=True)

    # Code if you want to load an already existing data set directly
    # Update possibility (was not changed to be consistent with existing experiment results):
        #Can be deleted
    """
    if path.exists(path_to_use['train_data']) and path.exists(path_to_use['train_label']) and path.exists(
            path_to_use['val_data']) and path.exists(path_to_use['val_label']) and path.exists(
            path_to_use['test_data']) and path.exists(path_to_use['test_label']):
        train_nn = np.load(path_to_use['train_data'])
        label_train_nn = np.load(path_to_use['train_label'])
        val = np.load(path_to_use['val_data'])
        label_val = np.load(path_to_use['val_label'])
        test = np.load(path_to_use['test_data'])
        label_test = np.load(path_to_use['test_label'])
        
    else:
    """

    # preprocess datasets
    train_nn, label_train_nn, val, label_val, test, label_test = prepare_dataset(size_train_nn, size_valid_nn,
                                                                                 dithering_used, one_against_all,
                                                                                 percent_of_major_label_to_keep=percent_of_major_label_to_keep,
                                                                                 number_class_to_predict=network.classes,
                                                                                 data_set_to_use=data_set_to_use, convert_to_grey = convert_to_grey)

    print('\n\n used data sets are saved')
    # save preprocessed datasets (needed for later evaluation)
    np.save(path_to_use['train_data'], train_nn)
    np.save(path_to_use['train_label'], label_train_nn)
    np.save(path_to_use['val_data'], val)
    np.save(path_to_use['val_label'], label_val)
    np.save(path_to_use['test_data'], test)
    np.save(path_to_use['test_label'], label_test)



    print("Start Training")
    # train network
    network.training(train_nn, label_train_nn, val, label_val, path_to_use)

    print("\n Start evaluate with train set ")
    # save accuracy on the NN on train set in results
    results.at[1, 'Neural network'] = network.evaluate(train_nn, label_train_nn)

    print("\n Start evaluate with validation set ")
    # should be the same value as the highest during training
    network.evaluate(val, label_val)

    print("\n Start evaluate with test set ")
    # save accuracy of the NN on test set in results
    results.at[3, 'Neural network'] = network.evaluate(test, label_test)

    print('end')
