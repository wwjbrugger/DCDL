"""
 train a dicision tree and to perform the sls algorithm with the inout and output_one_picture of a binary layer
"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"

import helper_methods as help
import numpy as np
import pickle



def SLS_Conv_1 (Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, SLS_Training, path_to_use):
    print('SLS Extraction for Convolution 1')

    data = np.load(path_to_use['input_conv_1'])
    label = np.load(path_to_use['label_conv_1'])
    used_kernel = np.load(path_to_use['g_kernel_conv_1'])

    path_to_store= path_to_use['logic_rules_conv_1']

    help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, data, label,
                                      used_kernel, result=None, path_to_store=path_to_store, SLS_Training=SLS_Training)

def prediction_Conv_1(path_to_use):
    print('Prediction with extracted rules for Convolution 1')

    data_flat = np.load(path_to_use['flat_data_conv_1'])
    label = np.load(path_to_use['g_sign_con_1'])
    logic_rule = pickle.load(open(path_to_use['logic_rules_conv_1'], "rb" ))

    path_to_store_prediction = path_to_use['prediction_conv_1']
    help.prediction_SLS_fast(data_flat, label, logic_rule, path_to_store_prediction)




def SLS_Conv_2 (Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, SLS_Training, Input_from_SLS, path_to_use):
    print('\n\n SLS Extraction for Convolution 2')
    print('Input Convolution 2', path_to_use['input_conv_2'])
    data = np.load(path_to_use['input_conv_2'])
    if Input_from_SLS:
        data = help.max_pooling(data)
    print('Label for Convolution 2: ', path_to_use['label_conv_2'])
    label = np.load(path_to_use['label_conv_2'])
    used_kernel = np.load(path_to_use['g_kernel_conv_2'])

    path_to_store= path_to_use['logic_rules_conv_2']

    help.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, data, label,
                                      used_kernel, result=None, path_to_store=path_to_store, SLS_Training = SLS_Training)



def prediction_Conv_2(path_to_use):
    print('Prediction with extracted rules for Convolution 2')

    data_flat = np.load(path_to_use['flat_data_conv_2'])
    label = np.load(path_to_use['label_conv_2'])
    found_formula = pickle.load(open(path_to_use['logic_rules_conv_2'], "rb"))

    path_to_store_prediction = path_to_use['prediction_conv_2']

    help.prediction_SLS_fast(data_flat, label, found_formula, path_to_store_prediction)

def SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, SLS_Training, path_to_use ):
    print('\n SLS Extraction for dense layer')
    print('data to use ', path_to_use['input_dense'] )
    data = np.load(path_to_use['input_dense'])
    label = np.load(path_to_use['label_dense'])
    if label.ndim == 1:
        label = label.reshape((-1, 1))

    path_to_store= path_to_use['logic_rules_dense']
    help.sls_dense_net(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, data, label,
                                       path_to_store=path_to_store, SLS_Training= SLS_Training)


def prediction_dense( path_to_use):
    print('\n  Prediction with extracted rules for dense layer')
    print('used label:', path_to_use['label_dense'])
    flat_data = np.load(path_to_use['flat_data_dense'])
    label = np.load(path_to_use['label_dense'])

    logic_rule = pickle.load(open(path_to_use['logic_rules_dense'], "rb"))
    path_to_store_prediction = path_to_use['prediction_dense']

    return help.prediction_SLS_fast(flat_data, label, logic_rule, path_to_store_prediction)


def visualize_kernel(one_against_all, path_to_kernel):
    print('Visualistation of the Kernel saved in {} is started '. format(path_to_kernel))
    kernel = np.load(path_to_kernel)
    if kernel.shape[2] >1:
        raise ValueError("Kernel which should be visualized has {} input channel visualization  is only for one channel implemented".format(kernel.shape[2]))
    for channel in range(kernel.shape[3]):
        help.visualize_singel_kernel(kernel[:,:,:,channel],kernel.shape[0] , "Kernel {} from {} for {} againt all \n  path: {}".format(channel, kernel.shape[3], one_against_all, path_to_kernel) )

if __name__ == '__main__':
    Number_of_disjuntion_term_in_SLS = 100
    Maximum_Steps_in_SKS = 10000
    stride_of_convolution = 2
    one_against_all = 2

   # visualize_kernel(one_against_all, 'data/kernel_conv_1.npy')

   # SLS_Conv_1(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution)

    #prediction_Conv_1()

    #SLS_Conv_2(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS,stride_of_convolution)

    #prediction_Conv_2()

    #SLS_dense(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS)
    prediction_dense()

