import time
from pathlib import Path
import numpy as np
import tensorflow as tf

import model.visualize_rules_found.one_conv_block_model as one_conv_block_model
import visualize_rules_found.train_network as first
import visualize_rules_found.data_generation as second
import visualize_rules_found.extract_logic_rules as third
import visualize_rules_found.reduce_kernel as fourths
import visualize_rules_found.helper_methods as help_vis
import model.Gradient_helpLayers_convBlock as helper_net


if __name__ == '__main__':
    default_store_path = Path('/home/jbrugger/PycharmProjects/dcdl_final/visualize_rules_found')
    settings_dic = {
        # set seed None if you don't want to set an explicit seed
        # seed is not working at the moment
        # Attention at the moment we can't set the seed for the SLS Algorithm
        'seed': None,
        'timestr': time.strftime("%Y%m%d-%H%M%S"),
        'default_store_path': default_store_path,
        'path_to_model_folder': default_store_path / 'model',
        # 'numbers' or 'fashion'
        'data_set_to_use': 'numbers',
        # size of train set
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

        # first part of name of nn model to train
        'name_of_model_prefix': 'visualize_rules_found',

        'learning_rate_net': 1E-3,
        'input_shape_net': (None, 28, 28, 1),
        'nr_train_iteration_net': 200,
        'batch_size_net': 2 ** 14,
        'print_train_acc_every_step_net': 100,
        'check_val_every_step_net': 10,
        'input_channels': 1,
        'activation': helper_net.binarize_STE,
        'use_bias_in_convolution_net': False,
        'number_of_kernel': 1,
        'k_interval': [150],
    }
    if settings_dic['seed'] is not None:
        np.random.seed(seed=settings_dic['seed'])
        tf.random.set_random_seed(seed= settings_dic['seed'])

    data_dic = {}
    for one_against_all in settings_dic['one_against_all_array']:
        # create parent folder, where to save model
        path_to_store_model_parent_folder = settings_dic['path_to_model_folder'] / str(one_against_all) / settings_dic['timestr']
        path_to_store_model_parent_folder.mkdir(parents=True, exist_ok=True)

        settings_dic['name_of_model'] = settings_dic['name_of_model_prefix'] + '_' + str(one_against_all)
        settings_dic['path_to_store_model'] = path_to_store_model_parent_folder / settings_dic['name_of_model']

        # get the neural net for datasets with one colour channel
        network = one_conv_block_model.network_one_convolution(
            name_of_model=settings_dic['name_of_model'],
            input_shape=settings_dic['input_shape_net'],
            batch_size=settings_dic['batch_size_net'],
            learning_rate=settings_dic['learning_rate_net'],
            shape_of_kernel=settings_dic['shape_of_kernel'],
            nr_training_iteration=settings_dic['nr_train_iteration_net'],
            stride=settings_dic['stride_of_convolution'],
            print_every=settings_dic['print_train_acc_every_step_net'],
            check_every=settings_dic['check_val_every_step_net'],
            number_of_kernel=settings_dic['number_of_kernel'],
            number_classes=settings_dic['number_classes_to_predict'],
            input_binarized=settings_dic['dithering_used'],
            activation=settings_dic['activation'],
            use_bias_in_convolution=settings_dic['use_bias_in_convolution_net'],
            input_channels=settings_dic['input_channels'],
            save_path_model=str(settings_dic['path_to_store_model']),
            save_path_logs=str(settings_dic['path_to_store_model'].parent))

        # train neural net
        data_dic = first.train_model(network=network,
                                     dithering_used=settings_dic['dithering_used'],
                                     one_against_all=one_against_all,
                                     data_set_to_use=settings_dic['data_set_to_use'],
                                     size_train_nn=settings_dic['size_train_nn'],
                                     size_valid_nn=settings_dic['size_valid_nn'],
                                     convert_to_grey=settings_dic['convert_to_grey'],
                                     values_max_1=settings_dic['pic_range_0_1'],
                                     data_dic=data_dic)

        # writes (intermediate) results to files, which are used for training and evaluation DCDL
        second.data_generation(network=network, data_dic=data_dic, data_type='train')

        help_vis.visualize_single_formula(
            kernel=data_dic['output_nn_kernel_conv_1_conv2d'],
            kernel_width=settings_dic['shape_of_kernel'][0],
            title="Kernel for {} against all".format(one_against_all),
            set_vmin_vmax=settings_dic['visualisation_range_-1_1'])

        for number_of_disjunction_term_in_SLS in settings_dic['k_interval']:
            # extract logical formula with SLS algorithm
            kernel_approximation_conv_1_conv2d = third.sls_convolution(
                training_set=data_dic['dither_train_data'],
                label_set=data_dic['output_nn_dcdl_conv_1_conv2d_Sign_train'],
                kernel_to_approximate=data_dic['output_nn_kernel_conv_1_conv2d'],
                number_of_disjunction_term_in_SLS=number_of_disjunction_term_in_SLS,
                Maximum_Steps_in_SKS=settings_dic['maximum_steps_in_SLS'],
                stride_of_convolution=settings_dic['stride_of_convolution'],
            )
            data_dic['kernel_approximation_conv_1_conv2d'] = kernel_approximation_conv_1_conv2d
            #  normalize logical rule for visualization
            rule_for_visualization = fourths.visualize_logic_rule(
                kernel_approximation=data_dic['kernel_approximation_conv_1_conv2d'],
                kernel_width=settings_dic['shape_of_kernel'][0],
                number_of_disjunction_term_in_SLS=number_of_disjunction_term_in_SLS,
                set_vmin_vmax=settings_dic['visualisation_range_-1_1'])
