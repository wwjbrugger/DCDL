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
        'logs': 'data/{}/logs/'.format(data_set_to_use),
        'store_model': 'data/{}/stored_models/'.format(data_set_to_use),
        'results': 'data/{}/results/'.format(data_set_to_use),

        'train_data': 'data/{}/train_data.npy'.format(data_set_to_use),
        'train_label': 'data/{}/train_label.npy'.format(data_set_to_use),
        'val_data': 'data/{}/val_data.npy'.format(data_set_to_use),
        'val_label': 'data/{}/val_label.npy'.format(data_set_to_use),
        'test_data': 'data/{}/test_data.npy'.format(data_set_to_use),
        'test_label': 'data/{}/test_label.npy'.format(data_set_to_use),

    }

    return path_to_use


def get_network(data_set_to_use, path_to_use, convert_to_gray):
    number_classes_to_predict = 2
    stride_of_convolution = 2
    shape_of_kernel = (2, 2)
    number_of_kernels = 8
    name_of_model = '{}_two_conv_2x2'.format(data_set_to_use)
    if data_set_to_use in 'mnist' or data_set_to_use in 'fashion':
        network = model_two_convolution.network_two_convolution(path_to_use, name_of_model=name_of_model,
                                                                shape_of_kernel=shape_of_kernel,
                                                                nr_training_itaration=2000,
                                                                stride=stride_of_convolution,
                                                                number_of_kernel=number_of_kernels,
                                                                number_classes=number_classes_to_predict)
    elif data_set_to_use in 'cifar':
        if convert_to_gray:
            input_channels = 1
            input_shape = (None, 32, 32, 1)
        else:
            input_channels = 3
            input_shape = (None, 32, 32, 3)
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
    column_name = []
    for name in methods_name:
        column_name.append(name+'_Train')
        column_name.append(name + '_Test')
    df = pd.DataFrame(index=[0], columns=column_name)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    return df



if __name__ == '__main__':
    if len(sys.argv) >= 2:
        print("used Dataset: ", sys.argv [1])
        print("Label-against-all", sys.argv [2])
        if (sys.argv[1] in 'mnist') or (sys.argv[1] in'fashion') or (sys.argv[1] in 'cifar'):
            data_set_to_use = sys.argv [1]
            one_against_all_list = [int(sys.argv [2])]
        else:
            raise ValueError('You choose a dataset which is not supported. \n Datasets which are allowed are mnist(Mnist), fashion(Fashion-Mnist) and cifar')
    else:
        data_set_to_use = 'mnist'  # 'mnist' or 'fashion'
        one_against_all_list = [5]#[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9,9,9]
    seed = None
    
    if data_set_to_use in 'mnist' or data_set_to_use in 'fashion': 
        size_train_nn = 55000
        size_valid_nn = 5000
        # train set is 10000
    elif data_set_to_use in 'cifar':
        size_train_nn = 45000
        size_valid_nn = 5000
        #train set is 10000

    for one_against_all in one_against_all_list: 
        dither_array = [ 'floyd-steinberg', 'atkinson', 'jarvis-judice-ninke', 'stucki', 'burkes',  'sierra-2-4a', 'stevenson-arce'] # 'sierra3',  'sierra2', 'atkinson', 'jarvis-judice-ninke', 'stucki', 'burkes',  'sierra-2-4a', 'stevenson-arce'
        dither_frame = get_pandas_frame_dither_method(dither_array)
        for dithering_used in dither_array:
            print('Dataset: {}, Label: {}, Dither Method: {}'.format(data_set_to_use, one_against_all, dithering_used))
            convert_to_grey = False

            path_to_use = get_paths(data_set_to_use)

            shape_of_kernel, stride_of_convolution, number_of_kernels, network = get_network(data_set_to_use, path_to_use,
                                                                                                 convert_to_grey)


            first.train_model(network=network, dithering_used=dithering_used, one_against_all=one_against_all,
                                      data_set_to_use=data_set_to_use, path_to_use=path_to_use, convert_to_grey=convert_to_grey,
                                      dither_frame=dither_frame, size_train_nn =size_train_nn, size_valid_nn= size_valid_nn, seed =seed)

        print(dither_frame)
        timestr = time.strftime("%Y%m%d-%H%M%S")
        path_to_store = 'dither_frames/' + '{}/label_{}__{}'.format(data_set_to_use, one_against_all, timestr)
        print('path_to_store_dither_frame ', path_to_store)
        dither_frame.to_pickle(path_to_store)
