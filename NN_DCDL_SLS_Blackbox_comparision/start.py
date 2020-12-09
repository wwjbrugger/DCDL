# This script compares the approaches neural net, DCDL and SLS with each other
# start script with python start.py []
import os
import argparse

os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"

import numpy as np
import tensorflow as tf
import SLS_Algorithm as SLS
import pickle as pickle
import pandas as pd
import get_data as get_data

import set_settings_cifar as settings_cifar
import set_settings_numbers_fashion as settings_number_fashion

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
        general_settings_dic, setting_dic_NN, settings_dic_SLS = \
            settings_cifar.get_experimental_settings()


    elif args.dataset in 'cifar':
        general_settings_dic, setting_dic_NN, settings_dic_SLS = \
            settings_number_fashion.get_experimental_settings()
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

    with open(path_to_store_settings, "wb") as f:
        pickle.dump({'general_settings_dic': general_settings_dic,
                     'setting_dic_NN': setting_dic_NN,
                     'settings_dic_SLS': settings_dic_SLS},
                    f)

    # --------------------create empty results frames and where to store them----------------------------------
    path_to_store_pd_results = \
        general_settings_dic['default_store_path'] / 'results/NN_DCDL_SLS_Blackbox_comparision{}.pkl'.format(
            general_settings_dic['timestr'])
    path_to_store_settings.parent.mkdir(parents=True, exist_ok=True)
    print('pandas results are stored in: ', path_to_store_pd_results)

    # create empty pandas files for results
    column_name = ['Neural network', 'DCDL', 'SLS BB prediction', 'SLS BB train']

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
            'number_classes_to_predict'],
        data_set_to_use=general_settings_dic['data_set_to_use'],
        convert_to_grey=general_settings_dic['convert_to_grey'],
        values_max_1=general_settings_dic['pic_range_0_1'],
        visualize_data = general_settings_dic['visualize_data'])

    train_neurel_net

