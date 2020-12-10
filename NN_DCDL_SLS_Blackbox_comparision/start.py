# This script compares the approaches neural net, DCDL and SLS with each other
# start script with python start.py []
import os
import argparse
from tabulate import tabulate

os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"

import numpy as np
import tensorflow as tf

import pickle as pickle
import pandas as pd
import get_data as get_data

import extract_data_from_neural_net as extract_data_from_neural_net

import set_settings_cifar as settings_cifar
import set_settings_numbers_fashion as settings_number_fashion

import NN_DCDL_SLS_Blackbox_comparision.Neural_net_model.NN_model as NN_model
import NN_DCDL_SLS_Blackbox_comparision.SLS_black_box_model.SLS_black_box as SLS_black_box_model
import NN_DCDL_SLS_Blackbox_comparision.DCDL_Model.DCDL as DCDL

if __name__ == '__main__':

    # -----------------------------------------parse command line arguments--------------------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset',
                        help='which dataset to load possible options are [numbers | fashion | cifar]',
                        type=str)
    parser.add_argument('one_against_all',
                        help='number of label \'one\' to run for \'one\' against all',
                        type=int)
    args = parser.parse_args()
    # -------------------------- get dictionaries with settings and store them ---------------
    if args.dataset in ('numbers' or 'fashion'):
        general_settings_dic, setting_dic_NN, \
        settings_dic_SLS_black_box_label, settings_dic_DCDL = \
            settings_number_fashion.get_experimental_settings()


    elif args.dataset in 'cifar':
        general_settings_dic, setting_dic_NN, \
        settings_dic_SLS_black_box_label, settings_dic_DCDL = \
            settings_cifar.get_experimental_settings()
    else:
        raise ValueError('chosen dataset [{}] is not supported '
                         'use start.py [-h] to get a list of supported datasets'.format(args.dataset))

    # dataset_to_use
    general_settings_dic['data_set_to_use'] = args.dataset
    # Label you want to test against all
    general_settings_dic['one_against_all'] = args.one_against_all

    # store settings
    path_to_store_settings = \
        general_settings_dic['default_store_path'] / 'settings/NN_DCDL_SLS_Blackbox_comparision{}.pkl'.format(
            general_settings_dic['timestr'])

    # create parent folder of place to store used settings
    path_to_store_settings.parent.mkdir(parents=True, exist_ok=True)
    print('settings are stored in: ', path_to_store_settings)

    # todo add other settings
    with open(path_to_store_settings, "wb") as f:
        pickle.dump({'general_settings_dic': general_settings_dic,
                     'setting_dic_NN': setting_dic_NN,
                     'settings_dic_SLS_black_box_label': settings_dic_SLS_black_box_label,
                     'settings_dic_DCDL':settings_dic_DCDL},
                    f)

    # --------------------where to store neural network model-----------------------------------
    path_to_store_model_parent_folder = \
        general_settings_dic['default_store_path'] / 'neural_net_saved_model' \
        / str(general_settings_dic['one_against_all']) / general_settings_dic['timestr']

    path_to_store_model_parent_folder.mkdir(parents=True, exist_ok=True)
    path_to_store_model = path_to_store_model_parent_folder / setting_dic_NN['name_of_model']
    # path to store the model of the neural net
    setting_dic_NN['save_path_model'] = str(path_to_store_model)
    setting_dic_NN['save_path_logs'] = str(path_to_store_model_parent_folder)

    # --------------------create empty results frames and where to store them----------------------------------
    path_to_store_pd_results = \
        general_settings_dic['default_store_path'] / 'results/NN_DCDL_SLS_Blackbox_comparision{}.pkl'.format(
            general_settings_dic['timestr'])
    path_to_store_settings.parent.mkdir(parents=True, exist_ok=True)
    print('pandas results are stored in: ', path_to_store_pd_results)

    # create empty pandas files for results
    column_name = ['Neural network', 'DCDL', 'SLS BB prediction', 'SLS BB label']

    results = pd.DataFrame(columns=column_name)

    # ---------------------------------------- set random seed if given-----------------------------------
    if general_settings_dic['seed'] is not None:
        np.random.seed(seed=general_settings_dic['seed'])
        tf.random.set_random_seed(seed=general_settings_dic['seed'])

    # -----------------------------------------------------------------------------------------------------
    # --------------------------------------START WITH EXPERIMENT------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    # ---------------------------------------- get_data ----------------------------------------------------
    data_dic = get_data.get_data(
        size_train=general_settings_dic['size_train'],
        size_valid=general_settings_dic['size_valid'],
        dithering_used=general_settings_dic['dithering_used'],
        one_against_all=general_settings_dic['one_against_all'],
        number_class_to_predict=general_settings_dic[
            'number_classes'],
        data_set_to_use=general_settings_dic['data_set_to_use'],
        convert_to_grey=general_settings_dic['convert_to_grey'],
        values_max_1=general_settings_dic['pic_range_0_1'],
        visualize_data=general_settings_dic['visualize_data'])


    # --------------------------------------train neural net-------------------------------------------------
    neural_net = NN_model.network_two_convolution(
        path_to_store_model=setting_dic_NN['save_path_model'],
        name_of_model=setting_dic_NN['name_of_model'],
        learning_rate=setting_dic_NN['learning_rate'],
        # length of one-hot-encoded label e.g.[0,1], after one_against_all
        number_classes=general_settings_dic['number_classes'],
        input_shape=general_settings_dic['shape_of_input_pictures'],
        nr_training_iteration=setting_dic_NN['number_train_iteration'],
        batch_size=setting_dic_NN['batch_size'],
        print_every=setting_dic_NN['print_acc_train_every'],
        check_every=setting_dic_NN['check_every'],
        number_of_kernel_conv_1=setting_dic_NN['num_kernel_conv_1'],
        number_of_kernel_conv_2=setting_dic_NN['num_kernel_conv_2'],
        shape_of_kernel_conv_1=setting_dic_NN['shape_of_kernel_conv_1'],
        shape_of_kernel_conv_2=setting_dic_NN['shape_of_kernel_conv_2'],
        stride_conv_1=setting_dic_NN['stride_of_convolution_conv_1'],
        stride_conv_2=setting_dic_NN['stride_of_convolution_conv_2'],
        input_channels=setting_dic_NN['input_channels'],
        # activation is a sign function sign(x) = -1 if x <= 0, 1 if x > 0.
        activation_str=setting_dic_NN['activation_str'],
        use_bias_in_conv_1=setting_dic_NN['use_bias_in_conv_1'],
        use_bias_in_conv_2=setting_dic_NN['use_bias_in_conv_2'],
        shape_max_pooling_layer=setting_dic_NN['shape_max_pooling_layer'],
        stride_max_pooling=setting_dic_NN['stride_max_pooling'],
        dropout_rate=setting_dic_NN['dropout_rate'],
        # use arg_min function to cast one hot label to true or false
        arg_min_label=general_settings_dic['arg_min_label'],
        logging=setting_dic_NN['logging'],
        save_path_logs=setting_dic_NN['save_path_logs'])

    neural_net.training(train=get_data.transform_boolean_to_minus_one_and_one(data_dic['train']),
                        label_train=data_dic['label_train'],
                        val=get_data.transform_boolean_to_minus_one_and_one(data_dic['val']),
                        label_val=data_dic['label_val'],
                        logging=setting_dic_NN['logging'])
    # --------------------------------------evaluate_neural_net-------------------------------------------------

    # save accuracy on the NN on train set in results
    results.at['Training set', 'Neural network'] = \
        neural_net.evaluate(input=get_data.transform_boolean_to_minus_one_and_one(data_dic['train']),
                            label=data_dic['label_train'])

    # should be the same value as the highest during training
    results.at['Validation set', 'Neural network'] = \
        neural_net.evaluate(input=get_data.transform_boolean_to_minus_one_and_one(data_dic['val']),
                            label=data_dic['label_val'])

    # save accuracy of the NN on test set in results
    results.at['Test set', 'Neural network'] = \
        neural_net.evaluate(input=get_data.transform_boolean_to_minus_one_and_one(data_dic['test']),
                            label=data_dic['label_test'])

    print('results after training neural net \n',
          tabulate(results.round(2), headers='keys', tablefmt='psql'))

    # ------------------------------------- train DCDL -----------------------------------------
    DCDL_data_dic = extract_data_from_neural_net.extract_data(
        neural_net=neural_net,
        input_neural_net=get_data.transform_boolean_to_minus_one_and_one(data_dic['train']),
        operations_in_DCDL=settings_dic_DCDL['operations'],
        print_nodes_in_neural_net=settings_dic_DCDL['print_nodes_in_neural_net'])

    DCDL_val_dic = extract_data_from_neural_net.extract_data(
        neural_net=neural_net,
        input_neural_net=get_data.transform_boolean_to_minus_one_and_one(data_dic['val']),
        operations_in_DCDL=settings_dic_DCDL['operations'],
        # already printed in step before
        print_nodes_in_neural_net=False)

    DCDL_objc = DCDL.DCDL(operations=settings_dic_DCDL['operations'],
                          arg_min_label=general_settings_dic['arg_min_label'])

    DCDL_objc.train(train_data=data_dic['train'],
                    validation_data=data_dic['val'],
                    use_prediction_operation_before=settings_dic_DCDL['use_prediction_operation_before'],
                    DCDL_data_dic=DCDL_data_dic,
                    DCDL_val_dic=DCDL_val_dic)

    # --------------------------------------evaluate DCDL - ------------------------------------------------
    results.at['Training set', 'DCDL'] = \
        DCDL_objc.prediction(data=data_dic['train'],
                             original_label=data_dic['label_train'])

    results.at['Validation set', 'DCDL'] = \
        DCDL_objc.prediction(data=data_dic['val'],
                             original_label=data_dic['label_val'])

    results.at['Test set', 'DCDL'] = \
        DCDL_objc.prediction(data=data_dic['test'],
                             original_label=data_dic['label_test'])

    print('results after training DCDL \n',
          tabulate(results.round(2), headers='keys', tablefmt='psql'))


    # ---------------------------------------train and evaluate SLS Blackbox with Label -----------------------------------
    SLS_black_box_label = SLS_black_box_model.SLS_black_box(mode=settings_dic_SLS_black_box_label['mode'],
                                                            arg_min_label=general_settings_dic['arg_min_label'],
                                                            number_of_disjunction_term=settings_dic_SLS_black_box_label[
                                                                'number_of_disjunction_term_in_SLS'],
                                                            maximum_steps_in_SLS=settings_dic_SLS_black_box_label[
                                                                'maximum_steps_in_SLS'],
                                                            init_with_kernel=settings_dic_SLS_black_box_label[
                                                                'init_with_kernel'],
                                                            p_g1=settings_dic_SLS_black_box_label['p_g1'],
                                                            p_g2=settings_dic_SLS_black_box_label['p_g2'],
                                                            p_s=settings_dic_SLS_black_box_label['p_s'],
                                                            batch=settings_dic_SLS_black_box_label['batch'],
                                                            cold_restart=settings_dic_SLS_black_box_label[
                                                                'cold_restart'],
                                                            decay=settings_dic_SLS_black_box_label['decay'],
                                                            min_prob=settings_dic_SLS_black_box_label['min_prob'],
                                                            zero_init=settings_dic_SLS_black_box_label['zero_init']
                                                            )
    SLS_black_box_label.train(train_data=data_dic['train'],
                              train_label=data_dic['label_train'],
                              validation_data=data_dic['val'],
                              validation_label=data_dic['label_val'])

    # save accuracy of the SLS_black_box_label on train set in results
    results.at['Training set', 'SLS BB label'] = \
        SLS_black_box_label.prediction(data=data_dic['train'],
                                       original_label=data_dic['label_train'])

    # save accuracy of the SLS_black_box_label on val set in results
    results.at['Validation set', 'SLS BB label'] = \
        SLS_black_box_label.prediction(data=data_dic['val'],
                                       original_label=data_dic['label_val'])

    # save accuracy of the NN on test set in results
    results.at['Test set', 'SLS BB label'] = \
        SLS_black_box_label.prediction(data=data_dic['test'],
                                       original_label=data_dic['label_test'])

    print('results after training SLS black box label \n',
          tabulate(results.round(2), headers='keys', tablefmt='psql'))





    with open(path_to_store_pd_results, "wb") as f:
        pickle.dump(results, f)
