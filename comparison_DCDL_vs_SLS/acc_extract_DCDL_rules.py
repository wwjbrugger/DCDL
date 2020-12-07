"""
 train a dicision tree and to perform the sls algorithm with the inout and output_one_picture of a binary layer
"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"

import helper_methods as helper_methods
import DCDL_helper as DCDL_helper
import numpy as np
import pickle



def DCDL_Conv_1 (number_of_disjunction_term_in_SLS_DCDL, maximum_steps_in_SLS_DCDL, stride_of_convolution, DCDL_train, path_to_use, unique_index):
    # load the data for training/using the DCDL rules for approximating the first convolution block.  
    print('SLS Extraction for Convolution 1')

    data = np.load(path_to_use['input_conv_1'])
    label = np.load(path_to_use['label_conv_1'])
    # load kernel of the neural net to get the shape of them 
    used_kernel = np.load(path_to_use['g_kernel_conv_1'])

    path_to_store= path_to_use['logic_rules_conv_1']
    
    DCDL_helper.sls_convolution(number_of_disjunction_term_in_SLS_DCDL = number_of_disjunction_term_in_SLS_DCDL,
                         maximum_steps_in_SLS_DCDL = maximum_steps_in_SLS_DCDL,
                         stride_of_convolution = stride_of_convolution,
                         data_sign = data,
                         label_sign = label,
                         used_kernel = used_kernel,
                         result=None,
                         path_to_store=path_to_store,
                         DCDL_train=DCDL_train,
                         unique_index =unique_index)

def prediction_DCDL_1(path_to_use):
    # use DCDL rules, which approximate the first convolution, for prediction 
    print('Prediction with extracted rules for Convolution 1')

    data_flat = np.load(path_to_use['flat_data_conv_1'])
    # label is loaded to create empty np array where the prediction is stored in 
    label = np.load(path_to_use['g_sign_con_1'])
    DCDL_logic_rule = pickle.load(open(path_to_use['logic_rules_conv_1'], "rb" ))

    path_to_store_prediction = path_to_use['prediction_conv_1']
    helper_methods.prediction_SLS_fast(data_flat =data_flat,
                             label=label,
                             found_formula =DCDL_logic_rule,
                             path_to_store_prediction= path_to_store_prediction)




def SLS_DCDL_2 (number_of_disjunction_term_in_SLS_DCDL, maximum_steps_in_SLS_DCDL, stride_of_convolution, DCDL_train, input_from_SLS, path_to_use, unique_index):
    # load the data for training/using the DCDL rules for approximating the second convolution block. 
    print('\n\n SLS Extraction for Convolution 2')
    print('Input Convolution 2', path_to_use['input_conv_2'])
    data = np.load(path_to_use['input_conv_2'])
    if input_from_SLS:
        # prediction of the DCDL approximation of convolution 1 is used. then we have to perform an max poolig step
        # used in experiment 
        # if data from neural net are used, this is not necessary, because this operation is allready done in the net.  
        data = helper_methods.max_pooling(data)
    # prediction of the neural net is used as label
    print('Label for Convolution 2: ', path_to_use['label_conv_2'])
    label = np.load(path_to_use['label_conv_2'])
    used_kernel = np.load(path_to_use['g_kernel_conv_2'])

    path_to_store= path_to_use['logic_rules_conv_2']
    # prepare data for learning/using DCDL rules to approximate convolutional layer
    DCDL_helper.sls_convolution(number_of_disjunction_term_in_SLS_DCDL=number_of_disjunction_term_in_SLS_DCDL,
                         maximum_steps_in_SLS_DCDL=maximum_steps_in_SLS_DCDL,
                         stride_of_convolution=stride_of_convolution,
                         data_sign=data,
                         label_sign=label,
                         used_kernel=used_kernel,
                         result=None,
                         path_to_store=path_to_store,
                         DCDL_train = DCDL_train,
                         unique_index=unique_index)



def prediction_DCDL_2(path_to_use):
    # use DCDL rules, which approximate the second convolution, for prediction 
    print('Prediction with extracted rules for Convolution 2')

    data_flat = np.load(path_to_use['flat_data_conv_2'])
    # label is loaded to create empty np array where the prediction is stored in
    label = np.load(path_to_use['label_conv_2'])
    found_formula = pickle.load(open(path_to_use['logic_rules_conv_2'], "rb"))

    path_to_store_prediction = path_to_use['prediction_conv_2']

    helper_methods.prediction_SLS_fast(data_flat=data_flat,
                             label=label,
                             found_formula=found_formula,
                             path_to_store_prediction=path_to_store_prediction)

def DCDL_dense(number_of_disjunction_term_in_SLS_DCDL, maximum_steps_in_SLS_DCDL, DCDL_train, path_to_use ):
    # load the data for training/using the DCDL rules for approximating the dense layer 
    print('\n SLS Extraction for dense layer')
    print('data to use ', path_to_use['input_dense'] )
    data = np.load(path_to_use['input_dense'])
    # depending on the path set in acc_main the prediction of the neural net,
    # the label of the train or the label of the test data is loaded 
    label = np.load(path_to_use['label_dense'])
    #if label.ndim == 1:
        # prediction from nn
   #     label = label.reshape((-1, 1))

    path_to_store= path_to_use['logic_rules_dense']
    DCDL_helper.sls_dense(number_of_disjunction_term_in_SLS_DCDL=number_of_disjunction_term_in_SLS_DCDL,
                   maximum_steps_in_SLS_DCDL=maximum_steps_in_SLS_DCDL,
                   data=data,
                   label=label,
                   path_to_store=path_to_store, SLS_Training= DCDL_train)


def prediction_dense( path_to_use):
    # use DCDL rules, which approximate the dense layer, for prediction 
    print('\n  Prediction with extracted rules for dense layer')
    print('used label:', path_to_use['label_dense'])
    
    flat_data = np.load(path_to_use['flat_data_dense'])
    # label is loaded to create empty np array where the prediction is stored and to calculate accuracy 
    label = np.load(path_to_use['label_dense'])
    if path_to_use['label_dense'] in path_to_use['train_label'] or path_to_use['label_dense'] in path_to_use['test_label'] :
        # cast One-hot-encoded label  [[0,1], [1,0], ...] to [-1,1, ...] as expected by SLS.calc_prediction_in_C
        # in comparison to prediction of the nn this is necessarily because the prediction of the neural net
        # is an arg_max operation if the data belong to class one it will return 0
        # if the data belong to class rest it will return 1
        label = np.array([np.argmax(one_hot_label) for one_hot_label in label])
    if path_to_use['label_dense'] in path_to_use['g_arg_max']:
        # label are prediction of the nn
        label = label.reshape((-1, 1))
        # cast prediction to boolean values [[-1],[1],[-1]] [[False], [True], [False]]
        label_set_one_hot = helper_methods.transform_to_boolean(label)
        # reduce two dimension shape to one dimensional shape
        label = np.array([label[0] for label in label_set_one_hot])

    logic_rule = pickle.load(open(path_to_use['logic_rules_dense'], "rb"))
    path_to_store_prediction = path_to_use['prediction_dense']

    # return accuracy of compared with the loaded data
    return helper_methods.prediction_SLS_fast(data_flat = flat_data,
                                    label = label,
                                    found_formula = logic_rule,
                                    path_to_store_prediction = path_to_store_prediction)



