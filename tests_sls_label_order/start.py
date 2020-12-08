# one-hot-label has the following form
# positive_label [1,0] instance is part of the one against all class
# inverse_label [1,0] instance is part of the rest class
# val means SLS is running with validation set
# no_val means SLS is running without validation set
# the values are the accuracy score on the train set
# Label one against all
# val_positive_label
# val_inverse_label
# no_val_positive_label
# no_val_inverse_label

import os

os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"

import numpy as np
import tensorflow as tf
import SLS_Algorithm as SLS
import pickle as pickle
import pandas as pd

import tests_sls_label_order.train_network as first
import tests_sls_label_order.helper_methods as helper_label_order
import tests_sls_label_order.set_settings as set_settings


def prepare_data_for_sls(data, label):
    # load data for sls approximation
    shape = data.shape

    # add colour channel to data
    data_flat = data.reshape((shape[0], - 1))
    label_flat = label[:, 0]
    # cast to True and False values
    training_set_flat = helper_label_order.transform_to_boolean(data_flat)
    label_flat = helper_label_order.transform_to_boolean(label_flat)
    return training_set_flat, label_flat

def calc_prediction(data, original_label, found_formula):

    num_input = data.shape[0]
    prediction = SLS.calc_prediction_in_C(data=data,
                                          label_shape=original_label.shape, found_formula=found_formula)
    error = np.sum(original_label != prediction)
    # calculate accuracy
    accuracy = (num_input - error) / num_input

    return accuracy


def get_data_for_SLS():
    train_nn, label_train_nn, \
    val_nn, label_val_nn, \
    test_nn, label_test_nn = first.prepare_dataset(size_train_nn=general_settings_dic['size_train_nn'],
                                                   size_valid_nn=general_settings_dic['size_valid_nn'],
                                                   dithering_used=general_settings_dic['dithering_used'],
                                                   one_against_all=one_against_all,
                                                   number_class_to_predict=general_settings_dic[
                                                       'number_classes_to_predict'],
                                                   data_set_to_use=general_settings_dic['data_set_to_use'],
                                                   convert_to_grey=general_settings_dic['convert_to_grey'],
                                                   values_max_1=general_settings_dic['pic_range_0_1'])

    # switch positions in one hot label e.g.: [0,1] -> [1,0]
    inverse_label_train_nn = np.array([[l[1], l[0]] for l in label_train_nn])
    inverse_label_val_nn = np.array([[l[1], l[0]] for l in label_val_nn])
    inverse_label_test_nn = np.array([[l[1], l[0]] for l in label_test_nn])

    # prepare_data_for_sls
    train, label_train = prepare_data_for_sls(data=train_nn, label=label_train_nn)
    val, label_val = prepare_data_for_sls(data=val_nn, label=label_val_nn)
    test, label_test = prepare_data_for_sls(data=test_nn, label=label_test_nn)

    _, inverse_label_train = prepare_data_for_sls(data=train_nn, label=inverse_label_train_nn)
    _, inverse_label_val = prepare_data_for_sls(data=val_nn, label=inverse_label_val_nn)
    _, inverse_label_test = prepare_data_for_sls(data=train_nn, label=inverse_label_test_nn)

    return train, val, test, \
           label_train, label_val, label_test, \
           inverse_label_train, inverse_label_val, inverse_label_test


if __name__ == '__main__':
    # get settings
    general_settings_dic, setting_dic_NN, settings_dic_SLS = set_settings.get_experimental_settings()

    # set random seed if given
    if general_settings_dic['seed'] is not None:
        np.random.seed(seed=general_settings_dic['seed'])
        tf.random.set_random_seed(seed=general_settings_dic['seed'])

    # store settings
    path_to_store_settings = general_settings_dic['default_store_path'] / 'settings/investigation_sls_{}.pkl'.format(
        general_settings_dic['timestr'])
    path_to_store_settings.parent.mkdir(parents=True, exist_ok=True)
    print('settings are stored in: ', path_to_store_settings)

    with open(path_to_store_settings, "wb") as f:
        pickle.dump({'general_settings_dic': general_settings_dic,
                     'setting_dic_NN': setting_dic_NN,
                     'settings_dic_SLS': settings_dic_SLS},
                    f)


    # pandas files for results
    column_name = ['val_positive_label', 'val_inverse_label',
                   'no_val_positive_label', 'no_val_inverse_label']

    results = pd.DataFrame(columns=column_name)

    # store settings
    path_to_store_pd_results = general_settings_dic['default_store_path'] / 'results/investigation_sls_{}.pkl'.format(
        general_settings_dic['timestr'])
    path_to_store_settings.parent.mkdir(parents=True, exist_ok=True)
    print('pandas results are stored in: ', path_to_store_pd_results)

    # start with experiment for different one_against_all label
    # try four different approaches
    #    - SLS without validation
    #    - SLS without validation inverted label
    #    - SLS with validation
    #    - SLS with validation inverted label
    for one_against_all in general_settings_dic['one_against_all_array']:
        print(20 * '-', one_against_all, 20 * '-')

        train, val, test, \
        label_train, label_val, label_test, \
        inverse_label_train, inverse_label_val, inverse_label_test = get_data_for_SLS()

        # SLS without validation and test and normal label
        found_formula_without_validation = SLS.rule_extraction_with_sls(
            train=train,
            train_label=label_train,
            number_of_disjunction_term=settings_dic_SLS[
                'number_of_disjunction_term_in_SLS'],
            maximum_steps_in_SLS=settings_dic_SLS['maximum_steps_in_SLS'],
            kernel=settings_dic_SLS['init_with_kernel'],
            p_g1=settings_dic_SLS['p_g1'],
            p_g2=settings_dic_SLS['p_g2'],
            p_s=settings_dic_SLS['p_s'],
            batch=settings_dic_SLS['batch'],
            cold_restart=settings_dic_SLS['cold_restart'],
            decay=settings_dic_SLS['decay'],
            min_prob=settings_dic_SLS['min_prob'],
            zero_init=settings_dic_SLS['zero_init']
        )
        # get acc at the validation set from training
        # validation set = train set in this case
        train_acc_without = found_formula_without_validation.train_acc
        # get acc from evaluating formula with train data
        prediction_acc_train_without =  calc_prediction(data=train,
                                                             original_label=label_train,
                                                             found_formula=found_formula_without_validation)
        print('no_val_positive_label: training {}  , prediction_train  {}  '.format(train_acc_without,
                                                                                  prediction_acc_train_without))
        np.testing.assert_equal(actual=train_acc_without,
                                desired=prediction_acc_train_without,
                                err_msg = 'Acc during training and when running in prediction mode \n'\
                                           'are different when running SLS without validation set \n'\
                                           'training without validation: {}'\
                                           'prediction for same data: {}'.format(train_acc_without,
                                                                                 prediction_acc_train_without))

        results.at[one_against_all, 'no_val_positive_label'] = \
            calc_prediction(data=test,
                            original_label=label_test,
                            found_formula=found_formula_without_validation)


        # SLS without validation and test  and inverse label
        found_formula_inverse_without_validation = SLS.rule_extraction_with_sls(
            train=train,
            train_label=inverse_label_train,
            number_of_disjunction_term=settings_dic_SLS[
                'number_of_disjunction_term_in_SLS'],
            maximum_steps_in_SLS=settings_dic_SLS[
                'maximum_steps_in_SLS'],
            kernel=settings_dic_SLS['init_with_kernel'],
            p_g1=settings_dic_SLS['p_g1'],
            p_g2=settings_dic_SLS['p_g2'],
            p_s=settings_dic_SLS['p_s'],
            batch=settings_dic_SLS['batch'],
            cold_restart=settings_dic_SLS['cold_restart'],
            decay=settings_dic_SLS['decay'],
            min_prob=settings_dic_SLS['min_prob'],
            zero_init=settings_dic_SLS['zero_init']
        )

        # get acc at the validation set from training
        # validation set = train set in this case
        train_acc_without_inverse = found_formula_inverse_without_validation.train_acc
        # get acc from evaluating formula with train data
        prediction_acc_train_without_inverse = calc_prediction(data=train,
                                                       original_label=inverse_label_train,
                                                       found_formula=found_formula_inverse_without_validation)
        print('no_val_positive_label: training {}  , prediction_train  {}  '.format(train_acc_without_inverse,
                                                                                    prediction_acc_train_without_inverse))

        np.testing.assert_equal(actual=train_acc_without_inverse,
                                desired=prediction_acc_train_without_inverse,
                                err_msg='Acc during training and when running in prediction mode \n' \
                                        'are different when running SLS without validation set \n' \
                                        'training without validation: {}' \
                                        'prediction for same data: {}'.format(train_acc_without_inverse,
                                                                              prediction_acc_train_without_inverse))

        results.at[one_against_all, 'no_val_inverse_label'] = \
            calc_prediction(data=test,
                            original_label=inverse_label_test,
                            found_formula=found_formula_inverse_without_validation)



        # SLS with validation and normal label
        found_formula_val = SLS.rule_extraction_with_sls_val(
            train=train,
            train_label=label_train,
            val=val,
            val_label=label_val,
            number_of_disjunction_term=settings_dic_SLS[
                'number_of_disjunction_term_in_SLS'],
            maximum_steps_in_SLS=settings_dic_SLS['maximum_steps_in_SLS'],
            kernel=settings_dic_SLS['init_with_kernel'],
            p_g1=settings_dic_SLS['p_g1'],
            p_g2=settings_dic_SLS['p_g2'],
            p_s=settings_dic_SLS['p_s'],
            batch=settings_dic_SLS['batch'],
            cold_restart=settings_dic_SLS['cold_restart'],
            decay=settings_dic_SLS['decay'],
            min_prob=settings_dic_SLS['min_prob'],
            zero_init=settings_dic_SLS['zero_init']
        )

        # get acc at the validation set from training
        # validation set = train set in this case
        train_acc_val = found_formula_val.train_acc
        # get acc from evaluating formula with train data
        prediction_acc_train_val = calc_prediction(data=val,
                                                               original_label=label_val,
                                                               found_formula=found_formula_val)
        print('no_val_positive_label: training {}  , prediction_train  {}  '.format(train_acc_val,
                                                                                    prediction_acc_train_val))

        np.testing.assert_equal(actual=train_acc_val,
                                desired=prediction_acc_train_val,
                                err_msg='val_positive_label \n '
                                        'Acc during training and when running in prediction mode \n' \
                                        'are different when running SLS  validation set \n' \
                                        'training validation: {}' \
                                        'prediction for same data: {}'.format(train_acc_val,
                                                                              prediction_acc_train_val))

        results.at[one_against_all, 'val_positive_label'] = \
            calc_prediction(data=test,
                            original_label=label_test,
                            found_formula=found_formula_val)

        # SLS with validation and inverse label
        found_formula_inverse_val = SLS.rule_extraction_with_sls_val(
            train=train,
            train_label=inverse_label_train,
            val=val,
            val_label=inverse_label_val,
            number_of_disjunction_term=settings_dic_SLS[
                'number_of_disjunction_term_in_SLS'],
            maximum_steps_in_SLS=settings_dic_SLS['maximum_steps_in_SLS'],
            kernel=settings_dic_SLS['init_with_kernel'],
            p_g1=settings_dic_SLS['p_g1'],
            p_g2=settings_dic_SLS['p_g2'],
            p_s=settings_dic_SLS['p_s'],
            batch=settings_dic_SLS['batch'],
            cold_restart=settings_dic_SLS['cold_restart'],
            decay=settings_dic_SLS['decay'],
            min_prob=settings_dic_SLS['min_prob'],
            zero_init=settings_dic_SLS['zero_init'],

        )

        # get acc at the validation set from training
        # validation set = train set in this case
        train_acc_val_inverse = found_formula_inverse_val.train_acc
        # get acc from evaluating formula with train data
        prediction_acc_train_val_inverse = calc_prediction(data=val,
                                                               original_label=inverse_label_val,
                                                               found_formula=found_formula_inverse_val)
        print('no_val_positive_label: training {}  , prediction_train  {}  '.format(train_acc_val_inverse,
                                                                                    prediction_acc_train_val_inverse))

        np.testing.assert_equal(actual=train_acc_val_inverse,
                                desired=prediction_acc_train_val_inverse,
                                err_msg='val_inverse_label\n'
                                        'Acc during training and when running in prediction mode \n' \
                                        'are different when running SLS validation set \n' \
                                        'training validation: {}' \
                                        'prediction for same data: {}'.format(train_acc_val_inverse,
                                                                              prediction_acc_train_val_inverse))

        results.at[one_against_all, 'val_inverse_label'] = \
            calc_prediction(data=test,
                            original_label=inverse_label_test,
                            found_formula=found_formula_inverse_val)

        with open(path_to_store_pd_results, "wb") as f:
            pickle.dump(results, f)
