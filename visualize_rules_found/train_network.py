import numpy as np

import comparison_DCDL_vs_SLS.acc_train as acc_train
import model.visualize_rules_found.one_conv_block_model as model_one_convolution
import comparison_DCDL_vs_SLS.data.numbers.mnist_dataset as numbers
import comparison_DCDL_vs_SLS.dithering as dith
import visualize_rules_found.helper_methods as help_vis

def prepare_dataset(size_train_nn, size_valid_nn, dithering_used=False, one_against_all=False, percent_of_major_label_to_keep = 0.1, number_class_to_predict = 10):
    print("Dataset processing", flush=True)
    dataset = numbers.data()
    dataset.get_iterator()

    train_nn, label_train_nn = dataset.get_chunk(size_train_nn)
    val, label_val = dataset.get_chunk(size_valid_nn)
    test, label_test = dataset.get_test()

    if train_nn.ndim == 3:
        train_nn = train_nn.reshape((train_nn.shape + (1,)))
        val = val.reshape((val.shape + (1,)))
        test = test.reshape((test.shape + (1,)))

    if dithering_used:
        train_nn = dith.dither_pic(train_nn)
        val = dith.dither_pic(val)
        test = dith.dither_pic(test)

    label_train_nn = help_vis.one_class_against_all(label_train_nn, one_against_all)
    label_val = help_vis.one_class_against_all(label_val, one_against_all )
    label_test = help_vis.one_class_against_all(label_test, one_against_all)

    train_nn, label_train_nn = acc_train.balance_data_set(train_nn, label_train_nn, percent_of_major_label_to_keep)
    val, label_val = acc_train.balance_data_set(val, label_val, percent_of_major_label_to_keep)
    test, label_test = acc_train.balance_data_set(test, label_test, percent_of_major_label_to_keep)
    print('size_of_trainingsset {}'.format(train_nn.shape[0]))
    return train_nn, label_train_nn, val, label_val, test, label_test



def train_model(network, dithering_used, one_against_all, number_classes_to_predict):
    #number_class_to_predict = 10
    dataset_to_use = 'numbers'
    size_train_nn = 55000
    size_valid_nn = 5000
    percent_of_major_label_to_keep = 0.111
    convert_to_grey = False

    train_nn, label_train_nn, val, label_val,  test, label_test = prepare_dataset(size_train_nn, size_valid_nn, dithering_used=dithering_used, one_against_all=one_against_all,
                    percent_of_major_label_to_keep=0.1, number_class_to_predict=number_classes_to_predict)
    print("Training", flush=True)
    #train_nn, label_train_nn, val, label_val,  test, label_test = comparison_DCDL_vs_SLS.train_network.prepare_dataset(size_train_nn, size_valid_nn, dithering_used, one_against_all,  data_set_to_use='mnist' )
    """train_nn, label_train_nn, val, label_val,  test, label_test = acc_train.prepare_dataset(size_train_nn =size_train_nn, size_valid_nn=size_valid_nn,
                    dithering_used= dithering_used , one_against_all = one_against_all,
                    percent_of_major_label_to_keep=percent_of_major_label_to_keep, number_class_to_predict=number_classes_to_predict, data_set_to_use=dataset_to_use,
                    convert_to_grey=convert_to_grey)
    """

    #prepare_dataset()
    #class_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    #dith.visualize_pic(train_nn, label_train_nn, class_names, "Input pic to train neuronal net with corresponding label", plt.cm.Greys)


    print("Start Training")
    network.training(train_nn, label_train_nn, test, label_test)

    print("\n Start evaluate with test set ")
    network.evaluate(test, label_test)
    print("\n Start evaluate with validation set ")
    network.evaluate(val, label_val)

    print ('\n\n used data sets are saved' )

    np.save('data/data_set_train.npy', train_nn)
    np.save('data/data_set_label_train_nn.npy', label_train_nn)
    np.save('data/data_set_val.npy', val)
    np.save('data/data_set_label_val.npy', label_val)
    np.save('data/data_set_test.npy', test)
    np.save('data/data_set_label_test.npy', label_test)

    print('end')

if __name__ == '__main__':
    dithering_used= True
    one_against_all = 4
    number_classes_to_predict = 2
    network = model_one_convolution.network_one_convolution(shape_of_kernel= (28,28), nr_training_itaration= 1000, stride=28, check_every= 200, number_of_kernel=1,
                                                            number_classes=number_classes_to_predict)
    train_model(network, dithering_used, one_against_all, number_classes_to_predict = number_classes_to_predict)