import os

os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import SLS_Algorithm as SLS

import tests_sls_label_order.train_network as first
import tests_sls_label_order.helper_methods as helper_lable_order


def call_sls_only_train_set(training_set, label_set, number_of_disjunction_term_in_SLS, maximum_steps_in_SLS,
                                init_with_kernel):
    # load data for sls approximation
    shape = training_set.shape

    # add colour channel to data
    training_set_flat = training_set.reshape((shape[0], - 1))
    label_set_flat = label_set[:, 0]
    # cast to True and False values
    training_set_flat = helper_lable_order.transform_to_boolean(training_set_flat)
    label_set_flat = helper_lable_order.transform_to_boolean(label_set_flat)

    # get formula for SLS blackbox approach
    found_formula = \
        SLS.rule_extraction_with_sls_without_validation_and_test(data=training_set_flat,
                                                        label=label_set_flat,
                                                        number_of_disjunction_term=number_of_disjunction_term_in_SLS,
                                                        maximum_steps_in_SLS=maximum_steps_in_SLS,
                                                        kernel=init_with_kernel)
    return found_formula


def call_sls_validation(training_set, label_set, number_of_disjunction_term_in_SLS, maximum_steps_in_SLS, init_with_kernel):
    # load data for sls approximation
    shape = training_set.shape

    # add colour channel to data
    training_set_flat = training_set.reshape((shape[0], - 1))
    label_set_flat = label_set[:, 0]
    # cast to True and False values
    training_set_flat = helper_lable_order.transform_to_boolean(training_set_flat)
    label_set_flat = helper_lable_order.transform_to_boolean(label_set_flat)

    # get formula for SLS blackbox approach
    found_formula = \
        SLS.rule_extraction_with_sls_without_test(data=training_set_flat,
                                     label=label_set_flat,
                                     number_of_disjunction_term=number_of_disjunction_term_in_SLS,
                                     maximum_steps_in_SLS=maximum_steps_in_SLS,
                                                  kernel=init_with_kernel)
    return found_formula


if __name__ == '__main__':
    default_store_path = Path('/home/jbrugger/PycharmProjects/dcdl_final/tests_sls_label_order')
    settings_dic = {
        # set seed None if you don't want to set an explicit seed
        # seed is not working at the moment
        # Attention at the moment we can't set the seed for the SLS Algorithm
        'seed': None,
        'timestr': time.strftime("%Y%m%d-%H%M%S"),
        'default_store_path': default_store_path,
        # 'numbers' or 'fashion'
        'data_set_to_use': 'numbers',

        'size_train_nn': 55000,

        # size of validation set
        'size_valid_nn': 5000,

        # length of one-hot-encoded label e.g.[0,1]
        'number_classes_to_predict': 2,

        # If pictures should be dithered set values to 0 or 1 (see https://en.wikipedia.org/wiki/Dither)
        'dithering_used': True,

        # Label you want to test against all
        'one_against_all_array': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],

        # are pixel of pictures in range [0 to 1]? other option [0 to 255]
        'pic_range_0_1': True,

        'maximum_steps_in_SLS': 2000,

        # stride use in convolution
        'stride_of_convolution': 28,

        # if pictures should be converted to grey before using
        'convert_to_grey': False,

        # shape of kernel used in convolution
        'shape_of_kernel': (28, 28),

        # for visualization of kernel and formula set vmin = -1 and vmax =1
        'visualisation_range_-1_1': True,

        'number_of_disjunction_term_in_SLS': 40,
        # if starts weights of the should be similar to kernel
        # if you want this you have to specify the kernel the weights should be similar to
        'init_with_kernel': False
    }
    # set random seed if given
    if settings_dic['seed'] is not None:
        np.random.seed(seed=settings_dic['seed'])
        tf.random.set_random_seed(seed=settings_dic['seed'])

    # write header in result file
    path_to_save_results = settings_dic['default_store_path'] / 'results/results_investigation_sls_{}.txt'.format(
        settings_dic['timestr'])

    with open(path_to_save_results, "w+") as file_object:
        file_object.write('one-hot-label has the following form \n')
        file_object.write('positive_label [1,0] instance is part of the one against all class\n')
        file_object.write('inverse_label [1,0] instance is part of the rest class\n')
        file_object.write('val means SLS is running with validation set ')
        file_object.write('no_val means SLS is running without validation set ')
        file_object.write('the values are the accuracy score on the train set\n' )
        file_object.write('Label one against all \t   val_positive_label  \t val_inverse_label'
                          ' \t   no_val_positive_label  \t no_val_inverse_label\n\n')

    # start with experiment for different one_against_all label
    # try four different approaches
    #    - SLS without validation
    #    - SLS without validation inverted label
    #    - SLS with validation
    #    - SLS with validation inverted label
    for one_against_all in settings_dic['one_against_all_array']:
        print(20*'-', one_against_all, 20*'-')
        train_nn, label_train_nn, \
        val, label_val, \
        test, label_test = first.prepare_dataset(size_train_nn=settings_dic['size_train_nn'],
                                                 size_valid_nn=settings_dic['size_valid_nn'],
                                                 dithering_used=settings_dic['dithering_used'],
                                                 one_against_all=one_against_all,
                                                 number_class_to_predict=settings_dic['number_classes_to_predict'],
                                                 data_set_to_use=settings_dic['data_set_to_use'],
                                                 convert_to_grey=settings_dic['convert_to_grey'],
                                                 values_max_1=settings_dic['pic_range_0_1'])

        # switch positions in one hot label e.g.: [0,1] -> [1,0]
        inverse_label_train_nn = np.array([[l[1], l[0]] for l in label_train_nn])

        # SLS with validation and normal label
        found_formula_val = call_sls_validation(training_set=train_nn,
                                                label_set=label_train_nn,
                                                number_of_disjunction_term_in_SLS=settings_dic[
                                                    'number_of_disjunction_term_in_SLS'],
                                                maximum_steps_in_SLS=settings_dic['maximum_steps_in_SLS'],
                                                init_with_kernel=settings_dic['init_with_kernel']
                                                )
        # SLS with validation and inverse label
        found_formula_inverse_val = call_sls_validation(training_set=train_nn,
                                                        label_set=inverse_label_train_nn,
                                                        number_of_disjunction_term_in_SLS=settings_dic[
                                                            'number_of_disjunction_term_in_SLS'],
                                                        maximum_steps_in_SLS=settings_dic['maximum_steps_in_SLS'],
                                                        init_with_kernel=settings_dic['init_with_kernel']
                                                        )
        # SLS without validation and test and normal label
        found_formula_without_validation = call_sls_only_train_set(training_set=train_nn,
                                                    label_set=label_train_nn,
                                                    number_of_disjunction_term_in_SLS=settings_dic[
                                                        'number_of_disjunction_term_in_SLS'],
                                                    maximum_steps_in_SLS=settings_dic['maximum_steps_in_SLS'],
                                                    init_with_kernel=settings_dic['init_with_kernel']
                                                    )

        # SLS without validation and test  and inverse label
        found_formula_inverse_without_validation = call_sls_only_train_set(training_set=train_nn,
                                                            label_set=inverse_label_train_nn,
                                                            number_of_disjunction_term_in_SLS=settings_dic[
                                                                'number_of_disjunction_term_in_SLS'],
                                                            maximum_steps_in_SLS=settings_dic['maximum_steps_in_SLS'],
                                                            init_with_kernel=settings_dic['init_with_kernel']
                                                            )

        result_string = '{0} \t\t\t\t\t  {1:.2f}  \t\t\t\t\t\t  {2:.2f}  \t\t\t\t\t  {3:.2f} \t\t\t\t\t  {4:.2f}  \n'.format(
            one_against_all, found_formula_val.train_acc, found_formula_inverse_val.train_acc,
            found_formula_without_validation.train_acc, found_formula_inverse_without_validation.train_acc)

        with open(path_to_save_results, "a") as file_object:
            file_object.write(result_string)
