"""
 train a dicision tree and to perform the sls algorithm with the inout and output_one_picture of a binary layer
"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"

import helper_methods as help
import numpy as np
import pickle



def SLS_Conv_1 (number_of_disjuntion_term_in_SLS_DCDL, maximum_steps_in_SLS_DCDL, stride_of_convolution, DCDL_train, path_to_use):
    # load the data for training/using the DCDL rules for approximating the first convolution block.  
    print('SLS Extraction for Convolution 1')

    data = np.load(path_to_use['input_conv_1'])
    label = np.load(path_to_use['label_conv_1'])
    # load kernel of the neural net to get the shape of them 
    used_kernel = np.load(path_to_use['g_kernel_conv_1'])

    path_to_store= path_to_use['logic_rules_conv_1']
    
    help.sls_convolution(number_of_disjuntion_term_in_SLS_DCDL, maximum_steps_in_SLS_DCDL, stride_of_convolution, data, label,
                                      used_kernel, result=None, path_to_store=path_to_store, SLS_Training=DCDL_train)

def prediction_Conv_1(path_to_use):
    # use DCDL rules, which approximate the first convolution, for prediction 
    print('Prediction with extracted rules for Convolution 1')

    data_flat = np.load(path_to_use['flat_data_conv_1'])
    # label is loaded to create empty np array where the prediction is stored in 
    label = np.load(path_to_use['g_sign_con_1'])
    logic_rule = pickle.load(open(path_to_use['logic_rules_conv_1'], "rb" ))

    path_to_store_prediction = path_to_use['prediction_conv_1']
    help.prediction_SLS_fast(data_flat, label, logic_rule, path_to_store_prediction)




def SLS_Conv_2 (number_of_disjuntion_term_in_SLS_DCDL, maximum_steps_in_SLS_DCDL, stride_of_convolution, DCDL_train, input_from_SLS, path_to_use):
    # load the data for training/using the DCDL rules for approximating the second convolution block. 
    print('\n\n SLS Extraction for Convolution 2')
    print('Input Convolution 2', path_to_use['input_conv_2'])
    data = np.load(path_to_use['input_conv_2'])
    if input_from_SLS:
        # prediction of the DCDL approximation of convolution 1 is used. then we have to perform an max poolig step
        # used in experiment 
        # if data from neural net are used, this is not necessary, because this operation is allready done in the net.  
        data = help.max_pooling(data)
    # prediction of the neural net is used as label
    print('Label for Convolution 2: ', path_to_use['label_conv_2'])
    label = np.load(path_to_use['label_conv_2'])
    used_kernel = np.load(path_to_use['g_kernel_conv_2'])

    path_to_store= path_to_use['logic_rules_conv_2']
    # prepare data for learning/using DCDL rules to approximate convolutional layer
    help.sls_convolution(number_of_disjuntion_term_in_SLS_DCDL, maximum_steps_in_SLS_DCDL, stride_of_convolution, data, label,
                                      used_kernel, result=None, path_to_store=path_to_store, SLS_Training = DCDL_train)



def prediction_Conv_2(path_to_use):
    # use DCDL rules, which approximate the second convolution, for prediction 
    print('Prediction with extracted rules for Convolution 2')

    data_flat = np.load(path_to_use['flat_data_conv_2'])
    # label is loaded to create empty np array where the prediction is stored in
    label = np.load(path_to_use['label_conv_2'])
    found_formula = pickle.load(open(path_to_use['logic_rules_conv_2'], "rb"))

    path_to_store_prediction = path_to_use['prediction_conv_2']

    help.prediction_SLS_fast(data_flat, label, found_formula, path_to_store_prediction)

def SLS_dense(number_of_disjuntion_term_in_SLS_DCDL, maximum_steps_in_SLS_DCDL, DCDL_train, path_to_use ):
    # load the data for training/using the DCDL rules for approximating the dense layer 
    print('\n SLS Extraction for dense layer')
    print('data to use ', path_to_use['input_dense'] )
    data = np.load(path_to_use['input_dense'])
    # depending on the path set in acc_main the prediction of the neural net,
    # the label of the train or the label of the test data is loaded 
    label = np.load(path_to_use['label_dense'])
    if label.ndim == 1:
        label = label.reshape((-1, 1))

    path_to_store= path_to_use['logic_rules_dense']
    help.sls_dense_net(number_of_disjuntion_term_in_SLS_DCDL, maximum_steps_in_SLS_DCDL, data, label,
                                       path_to_store=path_to_store, SLS_Training= DCDL_train)


def prediction_dense( path_to_use):
    # use DCDL rules, which approximate the dense layer, for prediction 
    print('\n  Prediction with extracted rules for dense layer')
    print('used label:', path_to_use['label_dense'])
    
    flat_data = np.load(path_to_use['flat_data_dense'])
    # label is loaded to create empty np array where the prediction is stored and to calculate accuracy 
    label = np.load(path_to_use['label_dense'])

    logic_rule = pickle.load(open(path_to_use['logic_rules_dense'], "rb"))
    path_to_store_prediction = path_to_use['prediction_dense']

    # return accuracy of compared with the loaded data
    return help.prediction_SLS_fast(flat_data, label, logic_rule, path_to_store_prediction)


def visualize_kernel(one_against_all, path_to_kernel):
    # plot images with the weights of the kernel
    # Update possibility (was not changed to be consistent with existing experiment results):
    # use 'Visualisation of the kernel saved in {} is started '
    print('Visualistation of the Kernel saved in {} is started '. format(path_to_kernel))
    kernel = np.load(path_to_kernel)
    if kernel.shape[2] >1:
        # kernel.shape -> [width, height, input_channel, output_channel]
        # Update possibility (was not changed to be consistent with existing experiment results):
        # "Kernel which should be visualized has {} input channel. Visualization is implemented only for one channel "
        raise ValueError("Kernel which should be visualized has {} input channel visualization  is only for one channel implemented".format(kernel.shape[2]))
    for channel in range(kernel.shape[3]):
        # iterate through the output channels of the kernels and visualize them
        help.visualize_singel_kernel(kernel[:,:,:,channel],kernel.shape[0] , "Kernel {} from {} for {} againt all \n  path: {}".format(channel, kernel.shape[3], one_against_all, path_to_kernel) )


# Update possibility (was not changed to be consistent with existing experiment results):
#    delete main method
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

