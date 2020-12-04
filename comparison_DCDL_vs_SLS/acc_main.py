import model.two_conv_block_model as model_two_convolution
import comparison_DCDL_vs_SLS.acc_train as first
import comparison_DCDL_vs_SLS.acc_data_generation as second
import comparison_DCDL_vs_SLS.acc_extracting_pictures as third
import sys
import pandas as pd
import comparison_DCDL_vs_SLS.sls_one_against_all as sls
import time



def get_paths(Input_from_SLS, use_label_predicted_from_nn, Training_set, data_set_to_use):
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

        'g_reshape': 'data/{}/data_reshape.npy'.format(data_set_to_use),
        'g_sign_con_1': 'data/{}/sign_con_1.npy'.format(data_set_to_use),
        'g_result_conv_1': 'data/{}/result_conv_1.npy'.format(data_set_to_use),
        'g_kernel_conv_1': 'data/{}/kernel_conv_1.npy'.format(data_set_to_use),
        'g_max_pool_1': 'data/{}/max_pool_1.npy'.format(data_set_to_use),
        'g_sign_con_2': 'data/{}/sign_con_2.npy'.format(data_set_to_use),
        'g_result_conv_2': 'data/{}/result_conv_2.npy'.format(data_set_to_use),
        'g_kernel_conv_2': 'data/{}/kernel_conv_2.npy'.format(data_set_to_use),
        'g_arg_max': 'data/{}/arg_max.npy'.format(data_set_to_use),

        'logic_rules_conv_1': 'data/{}/logic_rules_Conv_1'.format(data_set_to_use),
        'flat_data_conv_1': 'data/{}/logic_rules_Conv_1_data_flat.npy'.format(data_set_to_use),
        'prediction_conv_1': 'data/{}/prediction_for_conv_1.npy'.format(data_set_to_use),

        'logic_rules_conv_2': 'data/{}/logic_rules_Conv_2'.format(data_set_to_use),
        'flat_data_conv_2': 'data/{}/logic_rules_Conv_2_data_flat.npy'.format(data_set_to_use),
        'prediction_conv_2': 'data/{}/prediction_for_conv_2.npy'.format(data_set_to_use),

        'logic_rules_dense': 'data/{}/logic_rules_dense'.format(data_set_to_use),
        'flat_data_dense': 'data/{}/logic_rules_dense_data_flat.npy'.format(data_set_to_use),
        'prediction_dense': 'data/{}/prediction_dense.npy'.format(data_set_to_use),

        'logic_rules_SLS': 'data/{}/logic_rules_SLS'.format(data_set_to_use),

    }
    path_to_use['input_conv_1'] = path_to_use['g_reshape']
    path_to_use['label_conv_1'] = path_to_use['g_sign_con_1']
    path_to_use['label_conv_2'] = path_to_use['g_sign_con_2']

    if Training_set:
        path_to_use['input_graph'] = path_to_use['train_data']
    else:
        path_to_use['input_graph'] = path_to_use['test_data']

    if Input_from_SLS:
        path_to_use['input_conv_2'] = 'data/{}/prediction_for_conv_1.npy'.format(data_set_to_use)
        path_to_use['input_dense'] = 'data/{}/prediction_for_conv_2.npy'.format(data_set_to_use)
    else:
        path_to_use['input_conv_2'] = path_to_use['g_max_pool_1']
        path_to_use['input_dense'] = 'data/{}/sign_con_2.npy'.format(data_set_to_use)

    if use_label_predicted_from_nn and Training_set:
        path_to_use['label_dense'] = path_to_use['g_arg_max']
    elif use_label_predicted_from_nn and not Training_set:
        path_to_use['label_dense'] = path_to_use['g_arg_max']

    elif not use_label_predicted_from_nn and Training_set:
        path_to_use['label_dense'] = path_to_use['train_label']
    elif not use_label_predicted_from_nn and not Training_set:
        path_to_use['label_dense'] = path_to_use['test_label']

    return path_to_use


def get_network(data_set_to_use, path_to_use, convert_to_gray):
    number_classes_to_predict = 2
    stride_of_convolution = 2
    shape_of_kernel = (2, 2)
    number_of_kernels = 8
    name_of_model = '{}_two_conv_2x2'.format(data_set_to_use)
    if data_set_to_use in 'numbers' or data_set_to_use in 'fashion':
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
                                                                nr_training_itaration=2000,
                                                                stride=stride_of_convolution,
                                                                number_of_kernel=number_of_kernels,
                                                                number_classes=number_classes_to_predict,
                                                                input_channels=input_channels, input_shape=input_shape,
                                                                )

    return shape_of_kernel, stride_of_convolution, number_of_kernels, network

def get_pandas_frame (data_set_to_use, one_against_all):
    column_name = ['data_type', 'Used_label', 'Concat', 'SLS prediction', 'SLS train', 'Neural network']
    row_index = [0, 1, 2, 3]
    df = pd.DataFrame(index=row_index, columns=column_name)
    df.at[0, 'data_type'] = 'train'
    df.at[1, 'data_type'] = 'train'
    df.at[2, 'data_type'] = 'test'
    df.at[3, 'data_type'] = 'test'
    df.at[0, 'Used_label'] = 'Prediction_from_NN'
    df.at[1, 'Used_label'] = 'True_Label_of_Data'
    df.at[2, 'Used_label'] = 'Prediction_from_NN'
    df.at[3, 'Used_label'] = 'True_Label_of_Data'
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    return df

def fill_data_frame_with_concat(results, result_concat, use_label_predicted_from_nn, Training_set ):
    if use_label_predicted_from_nn and Training_set:
        results.at[0,'Concat'] = result_concat
    if not use_label_predicted_from_nn and Training_set:
        results.at[1,'Concat'] = result_concat
    if use_label_predicted_from_nn and not Training_set:
        results.at[2,'Concat'] = result_concat
    if not use_label_predicted_from_nn and not Training_set:
        results.at[3,'Concat'] = result_concat


def fill_data_frame_with_sls(results, result_SLS_train, result_SLS_test, use_label_predicted_from_nn):
    if use_label_predicted_from_nn:
        results.at[0, 'SLS prediction'] = result_SLS_train
        results.at[3, 'SLS prediction'] = result_SLS_test
    else:
        results.at[1, 'SLS train'] = result_SLS_train
        results.at[3, 'SLS train'] = result_SLS_test



if __name__ == '__main__':
    if len(sys.argv) > 1:

        print("used Dataset: ", sys.argv [1])
        print("Label-against-all", sys.argv [2])
        if (sys.argv[1] in 'numbers') or (sys.argv[1] in'fashion') or (sys.argv[1] in 'cifar'):
            data_set_to_use = sys.argv [1]
            one_against_all = int(sys.argv [2])
        else:
            raise ValueError('You choose a dataset which is not supported. \n Datasets which are allowed are numbers(Mnist), fashion(Fashion-Mnist) and cifar')
    else:
        data_set_to_use = 'cifar'  # 'numbers' or 'fashion'
        one_against_all = 4



    dithering_used = True
    convert_to_grey = False
    NN_Train_l = [True, False, False, False]
    SLS_Training_l = [True, False, False, False]  # Should SLS generate Rules
    Training_set_l = [True, True, False, False]  # Should Trainingset be used or test set
    use_label_predicted_from_nn_l = [True, False, True,
                                     False]  # for prediction in last layer should the output of the nn be used or true label
    Input_from_SLS = True  # for extracting the rules ahould the input be the label previously calculated by SLS
    mode = ['train data prediction', 'train data true label', 'test data prediction', 'test data true label']


    Number_of_disjuntion_term_in_SLS = 40
    Maximum_Steps_in_SKS = 2000

    results = get_pandas_frame(data_set_to_use, one_against_all)

    for i in range(len(NN_Train_l)):
        NN_Train = NN_Train_l[i]
        SLS_Training = SLS_Training_l[i]
        Training_set = Training_set_l[i]
        use_label_predicted_from_nn = use_label_predicted_from_nn_l[i]

        path_to_use = get_paths(Input_from_SLS, use_label_predicted_from_nn, Training_set, data_set_to_use)

        shape_of_kernel, stride_of_convolution, number_of_kernels, network = get_network(data_set_to_use, path_to_use,
                                                                                         convert_to_grey)

        if NN_Train:
            first.train_model(network, dithering_used, one_against_all, data_set_to_use, path_to_use, convert_to_grey, results)
        print('\n\n\n\t\t\t', mode[i])
        second.acc_data_generation(network, path_to_use)

        # third.visualize_kernel(one_against_all, 'data/kernel_conv_1.npy')

        third.SLS_Conv_1(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, SLS_Training,
                         path_to_use)

        third.prediction_Conv_1(path_to_use)

        third.SLS_Conv_2(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, SLS_Training,
                         Input_from_SLS, path_to_use)

        third.prediction_Conv_2(path_to_use)

        third.SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, SLS_Training, path_to_use)

        result_concat = third.prediction_dense(path_to_use)

        if Training_set:  # once for output of nn and once for true data
            found_formula, result_SLS_train = sls.SLS_on_diter_data_against_true_label(path_to_use)
            result_SLS_test = sls.predicition(found_formula, path_to_use)
            fill_data_frame_with_sls(results, result_SLS_train, result_SLS_test, use_label_predicted_from_nn)

        fill_data_frame_with_concat(results, result_concat, use_label_predicted_from_nn, Training_set )
    print(results, flush=True)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    print('path_to_store_results' , path_to_use['results']+ 'label_{}__{}'.format(one_against_all, timestr ))
    results.to_pickle(path_to_use['results']+ 'label_{}__{}'.format(one_against_all, timestr))
