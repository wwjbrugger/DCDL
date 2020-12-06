import model.two_conv_block_model as model_two_convolution
import compare_dither.train_network as first

import sys
import pandas as pd
import time



def get_paths(data_set_to_use):
    """
    Paths to save intermediate and final results 
    @param input_from_SLS: 
    @param use_label_predicted_from_nn: 
    @param training_set: 
    @param data_set_to_use: 
    @return: 
    """
    path_to_use = {
        # where the logs of the  neural net are saved
        'logs': 'data/{}/logs/'.format(data_set_to_use),
        # where the trained model is stored
        'store_model': 'data/{}/stored_models/'.format(data_set_to_use),
        # where the pandas frames are stored with the results of the dithering
        'results': 'data/{}/results/'.format(data_set_to_use),

        # where the data with which the neural net is trained are stored
        'train_data': 'data/{}/train_data.npy'.format(data_set_to_use),
        # where the label with which the neural net is trained are stored
        'train_label': 'data/{}/train_label.npy'.format(data_set_to_use),
        # where the validation data with which the neural net is chosen are stored
        'val_data': 'data/{}/val_data.npy'.format(data_set_to_use),
        # where the validation label with which the neural net is chosen are stored
        'val_label': 'data/{}/val_label.npy'.format(data_set_to_use),
        # where the data with which the neural net is evaluated are stored
        'test_data': 'data/{}/test_data.npy'.format(data_set_to_use),
        # where the data label which the neural net is evaluated are stored
        'test_label': 'data/{}/test_label.npy'.format(data_set_to_use),

    }

    return path_to_use


def get_network(data_set_to_use, path_to_use, convert_to_gray):
    # get network with two convolution and one dense layer at the end
    # net for dataset 'numbers' (MNIST) and 'fashion' (Fashion-MNIST)
    # have one colour channel as input writen in from [num_pic, pic_width, pic_height, num_colour_channel]
    # net for dataset 'cifar' (CIFAR)
    # have there colour channel as input writen in from [num_pic, pic_width, pic_height, num_colour_channel]

    # how many classes we want to distinguish, we are using one-against-all_testing so there are 2 classes
    number_classes_to_predict = 2
    # stride use in convolution
    stride_of_convolution = 2
    # shape of kernel used in convolution
    shape_of_kernel = (2, 2)
    # how many kernels are used in convolution. Every kernel is approximated by a logical formula.
    number_of_kernels = 8
    # name under which the model is stored after training
    name_of_model = '{}_two_conv_2x2'.format(data_set_to_use)
    if data_set_to_use in 'mnist' or data_set_to_use in 'fashion':
        network = model_two_convolution.network_two_convolution(path_to_use, name_of_model=name_of_model,
                                                                shape_of_kernel=shape_of_kernel,
                                                                nr_training_itaration=2000,
                                                                stride=stride_of_convolution,
                                                                number_of_kernel=number_of_kernels,
                                                                number_classes=number_classes_to_predict)
    elif data_set_to_use in 'cifar':
        # dataset will be first converted to gray scale picture.
        # After this the dither operation is applied on the grey scale picture.
        # Not used in final run
        if convert_to_gray:
            input_channels = 1
            input_shape = (None, 32, 32, 1)
        else:
            input_channels = 3
            input_shape = (None, 32, 32, 3)
        # get the neural net for datasets with three colour channel
        network = model_two_convolution.network_two_convolution(path_to_use, name_of_model=name_of_model,
                                                                shape_of_kernel=shape_of_kernel,
                                                                nr_training_itaration=2500,
                                                                stride=stride_of_convolution,
                                                                number_of_kernel=number_of_kernels,
                                                                number_classes=number_classes_to_predict,
                                                                input_channels=input_channels, input_shape=input_shape,
                                                                )

    return shape_of_kernel, stride_of_convolution, number_of_kernels, network

def get_pandas_frame_dither_method (methods_name):
    """
    Place to save the performance of the neural net for the different used dither methods.
    @param methods_name: 
    @return: 
    """
    #            | Method_0_train | Method_0_test | Method_1_train | Method_0_test network
    #  accuracy  |       X        |        X      |      X         |           X

    column_name = []
    for name in methods_name:
        column_name.append(name+'_Train')
        column_name.append(name + '_Test')
    df = pd.DataFrame(index=[0], columns=column_name)

    # change setting of pandas frame to be shown without cutting in terminal
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    return df



if __name__ == '__main__':
    # start script with python start.py to use parameter set in script
    # start script with python start.py [dataset] [label for one against all]
    #   e.g. python start.py fashion 6
    # will run the experiment for the Fashion Mnist Dataset with label 6 against all.
    if len(sys.argv) >= 2:
        print("used Dataset: ", sys.argv [1])
        print("Label-against-all", sys.argv [2])
        if (sys.argv[1] in 'mnist') or (sys.argv[1] in'fashion') or (sys.argv[1] in 'cifar'):
            data_set_to_use = sys.argv [1]
            one_against_all_list = [int(sys.argv [2])]
        else:
            raise ValueError('You choose a dataset which is not supported. \n Datasets which are allowed are mnist(Mnist), fashion(Fashion-Mnist) and cifar(CIFAR)')
    else:
        # values if you start script without parameter
        data_set_to_use = 'mnist'  # 'mnist' or 'fashion' or 'cifar'
        one_against_all_list = [5]#[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9]

    # seed for sampling dataset
    # update possibility (was not changed to be consistent with existing experiment results):
    #   use seed for training neural net
    seed = None
    
    if data_set_to_use in 'mnist' or data_set_to_use in 'fashion':
        # size of data set to train nn
        size_train_nn = 55000
        # size of data set to validate nn
        size_valid_nn = 5000
        # train set is 10000
    elif data_set_to_use in 'cifar':
        # size of data set to train nn
        size_train_nn = 45000
        # size of data set to validate nn
        size_valid_nn = 5000
        #train set is 10000

    for one_against_all in one_against_all_list:
        # possible dither algorithms:
        # [ 'floyd-steinberg', 'atkinson', 'jarvis-judice-ninke', 'stucki', 'burkes',  'sierra-2-4a', 'stevenson-arce', 'sierra3',  'sierra2', 'atkinson', 'jarvis-judice-ninke', 'stucki', 'burkes',  'sierra-2-4a', 'stevenson-arce']
        dither_array = [ 'floyd-steinberg', 'atkinson', 'jarvis-judice-ninke', 'stucki', 'burkes',  'sierra-2-4a', 'stevenson-arce']
        dither_frame = get_pandas_frame_dither_method(dither_array)
        for dithering_used in dither_array:
            print('Dataset: {}, Label: {}, Dither Method: {}'.format(data_set_to_use, one_against_all, dithering_used))
            convert_to_grey = False

            path_to_use = get_paths(data_set_to_use)

            shape_of_kernel, stride_of_convolution, number_of_kernels, network = get_network(data_set_to_use, path_to_use,
                                                                                                 convert_to_grey)

            # train nn with dithered dataset and evaluate it on dithered test set
            # write accuracy values into dither frame
            first.train_model(network=network, dithering_used=dithering_used, one_against_all=one_against_all,
                                      data_set_to_use=data_set_to_use, path_to_use=path_to_use, convert_to_grey=convert_to_grey,
                                      dither_frame=dither_frame, size_train_nn =size_train_nn, size_valid_nn= size_valid_nn, seed =seed)

        # print and store result frame
        print(dither_frame)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path_to_store = 'dither_frames/' + '{}/label_{}__{}'.format(data_set_to_use, one_against_all, timestr)
        print('path_to_store_dither_frame ', path_to_store)
        dither_frame.to_pickle(path_to_store)
