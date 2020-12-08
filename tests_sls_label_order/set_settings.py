import time
from pathlib import Path

def get_experimental_settings():
    default_store_path = Path('/home/jbrugger/PycharmProjects/dcdl_final/tests_sls_label_order')

    general_settings_dic = {
        # set seed None if you don't want to set an explicit seed
        # seed is not working at the moment
        # Attention at the moment we can't set the seed for the SLS Algorithm
        'seed': None,
        'timestr': time.strftime("%Y%m%d-%H%M%S"),
        'default_store_path': default_store_path,
        # 'numbers' or 'fashion'
        'data_set_to_use': 'numbers',

        # for numbers and fashion should the value be 55000
        # for cifar 45000
        #todo set back to 55000
        'size_train_nn': 4000,


        # size of validation set
        'size_valid_nn': 5000,

        # If pictures should be dithered set values to 0 or 1 (see https://en.wikipedia.org/wiki/Dither)
        'dithering_used': True,

        # Label you want to test against all
        'one_against_all_array': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],

        # length of one-hot-encoded label e.g.[0,1], after one_against_all
        'number_classes_to_predict': 2,

        # are pixel of pictures in range [0 to 1]? other option [0 to 255]
        'pic_range_0_1': True,


        # if pictures should be converted to grey before using
        'convert_to_grey': False,

        # for visualization of kernel and formula set vmin = -1 and vmax =1
        'visualisation_range_-1_1': True,
    }

    setting_dic_NN = {
        # shape of kernel used in convolution
        'shape_of_kernel': (28, 28),
        # stride use in convolution
        'stride_of_convolution': 28,

    }

    settings_dic_SLS = {
        # which split of data is used for finding logical rules:
        # 'train' -> all data are in train set
        # train_val -> data are split in train and validation set
        # train_val_test -> data are split in train, validation and test set () more interesting for
        # mode should be set in experiment
        'mode': None ,
        # number_of_disjunction_term_in_SLS
        'number_of_disjunction_term_in_SLS': 40,
        # todo set back to 2000
        'maximum_steps_in_SLS': 200,

        # Probability in SLS to choose a term uniformly drown from formula
        # if missed_instance has positive label
        # otherwise the SLS search for the formula
        # which differs in the smallest number of literals from missed_instance
        'p_g1': .5,

        # Probability in SLS to choose a literal uniformly drown from term
        # if missed_instance has positive label.
        # this literal will be removed from term
        # otherwise the SLS removes all literals in term that differ from missed_instance
        'p_g2': .5,

        # Probability in SLS to choose a literal uniformly drown from formula
        # if missed_instance has negative label
        # otherwise uniformly pick 1024 many training instances
        # search for literal whose addition to terms reduces score(batch) the most
        # This instance will be added to term, which covers missed_instance before
        'p_s': .5,

        # if starts weights of the should be similar to kernel
        # if you want this you have to specify the kernel the weights should be similar to
        'init_with_kernel': False,

        # use batch in SLS
        'batch' : True,

        # if no more improvement found for 600 steps restart SLS with new random formula
        'cold_restart' : True,

        # Decay factor, with which p_g1 and p_s can be reduces after train step be zero.
        # Up to min_prob
        'decay' : 0,
        'min_prob': 0,
        # initialize SLS with empty formula instead of random
        'zero_init' : False,
    }
    return general_settings_dic, setting_dic_NN, settings_dic_SLS
