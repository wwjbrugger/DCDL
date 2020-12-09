import time
from pathlib import Path


def get_experimental_settings():
    #todo fit to DCDL experiment
    default_store_path = Path('/home/jbrugger/PycharmProjects/dcdl_final/NN_DCDL_SLS_Blackbox_comparision')

    general_settings_dic = {
        # set seed None if you don't want to set an explicit seed
        # seed is not working at the moment
        # Attention at the moment we can't set the seed for the SLS Algorithm
        'seed': None,
        'timestr': time.strftime("%Y%m%d-%H%M%S"),
        'default_store_path': default_store_path,


        # for numbers and fashion should the value be 55000
        'size_train': 55000,


        # size of validation set
        'size_valid': 5000,

        # shape of input pictures
        'shape_of_input_pictures' : [28,28],

        # If pictures should be dithered set values to 0 or 1 (see https://en.wikipedia.org/wiki/Dither)
        'dithering_used': True,

        # length of one-hot-encoded label e.g.[0,1], after one_against_all
        'number_classes': 2,

        # are pixel of pictures in range [0 to 1]? other option [0 to 255]
        'pic_range_0_1': True,


        # if pictures should be converted to grey before using

        'convert_to_grey': False,

        # for visualization of kernel and formula set vmin = -1 and vmax =1
        'visualisation_range_-1_1': True,

        # visualize data during training
        'visualize_data': True,

        # how to convert [x,y] to one number switch meaning of label
        # e.g. arg_min([0,1]) = 0
        # e.g. arg_max([0, 1]) = 1
        'arg_min_label': True
    }


    setting_dic_NN = {
        # name_of_model
        'name_of_model': 'two_conv_2x2_{}'.format(general_settings_dic['timestr']),
        # number_train_iteration
        'number_train_iteration':2000,
        # shape of kernel used in first convolution
        'shape_of_kernel_conv_1': (2, 2),
        # number kernel used in first_convolution
        'num_kernel_conv_1': 8 ,
        # shape of kernel used in second convolution
        'shape_of_kernel_conv_2': (2, 2),
        # number kernel used in first_convolution
        'num_kernel_conv_2' :8,
        # stride use in convolution
        'stride_of_convolution_conv_1': 2,
        # stride use in convolution
        'stride_of_convolution_conv_2': 2,
        # number elements in batch
        'batch_size': 2**10,
        # after how many steps should acc of the the train print again
        'print_acc_train_every' :1000,
        # after how many steps should the NN be validated
        # net with highest validation score will be used at the end.
        'check_every' :25,
        # use bias in convolution 1
        'use_bias_in_conv_1' : False,
        # use bias in convolution 1
        'use_bias_in_conv_2' : False,
        # activation after conv 1 is only 'binarize_STE' and 'relu'
        'activation_str' : 'binarize_STE',
        # shape_max_pooling_layer which parts of the input should be pooled
        'shape_max_pooling_layer': [1,2,2,1],
        # stride of max pooling layer
        'stride_max_pooling': 2,
        # learning rate net
        'learning_rate':1E-3,
        # dropout rate
        'dropout_rate': 0.4,
        # use logging mode while training the net
        'loging':True,


    }
    # input channel for numbers and fashion is always 1
    # code is for consistency with  cifar set_settings
    if general_settings_dic['convert_to_grey']:
        # pic are cast to grey scale pic
        setting_dic_NN['input_channels'] = 1
    else:
        # pics have all three colour channel
        setting_dic_NN['input_channels'] = 1

    settings_dic_SLS = {
        # which split of data is used for finding logical rules:
        # 'train' -> all data are in train set
        # train_val -> data are split in train and validation set
        # train_val_test -> data are split in train, validation and test set () more interesting for
        # mode should be set in experiment
        'mode': None ,
        # number_of_disjunction_term_in_SLS
        'number_of_disjunction_term_in_SLS': 40,

        'maximum_steps_in_SLS': 2000,

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
