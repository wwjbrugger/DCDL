"""
 train a dicision tree and to perform the sls algorithm with the inout and output_one_picture of a binary layer
"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import helper_methods as helper_methods
import numpy as np
import SLS_Algorithm as SLS
import pickle

def sls_convolution ( number_of_disjunction_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, one_against_all) :

    label_set = np.load('data/label_SLS.npy')
    training_set = np.load('data/data_set_train.npy')
    shape = training_set.shape
    result_conv = np.load('data/result_conv.npy')
    kernel = np.load('data/kernel.npy')

    training_set=training_set.reshape((shape[0],shape[1],shape[2],1))
    training_set = helper_methods.transform_to_boolean(training_set)
    label_set = helper_methods.transform_to_boolean(label_set)
    kernel_width = kernel.shape[0]
    values_under_kernel = helper_methods.data_in_kernel(training_set, stepsize=stride_of_convolution, width=kernel_width)

    kernel_approximation = []

    for channel in range(label_set.shape[3]):
        print("Ruleextraction for Kernel {} ".format(channel))
        training_set_flat, label_set_flat = helper_methods.permutate_and_flaten(values_under_kernel, label_set,
                                                                      channel_label=channel)
        found_formula = \
            SLS.rule_extraction_with_sls_without_validation(training_set_flat, label_set_flat,  number_of_disjunction_term_in_SLS,
                                                            Maximum_Steps_in_SKS, np.sign(np.reshape(kernel[:,:,:,channel], -1)))
        kernel_approximation.append(found_formula)

        #label_self_calculated = helper_methods.calculate_convolution(values_under_kernel, kernel[:, :, :, channel], result_conv)
    pickle.dump(kernel_approximation, open('data/logic_formula_label_{}'.format(one_against_all), "wb"))

    formel_in_array_code = []
    for formel in kernel_approximation:
        formel_in_array_code.append(np.reshape(formel.formula_in_arrays_code, (-1, kernel_width, kernel_width)))
    np.save('data/kernel_approximation_label_{}.npy'.format(one_against_all), formel_in_array_code)
    return 


def visualize_kernel(one_against_all):
    print('Visualistation of the Kernel is started ')
    kernel = np.load('data/kernel.npy')

    label_for_pic = ['kernel {} '.format(i) for i in range(kernel.shape[3])]

    helper_methods.visualize_singel_kernel(kernel, 28, "Kernel for {} againt all".format(one_against_all), set_vmin_vmax = True)

if __name__ == '__main__':
    Number_of_disjuntion_term_in_SLS = 4
    Maximum_Steps_in_SKS = 100
    stride_of_convolution = 28
    one_against_all = 2

    sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution,one_against_all)


