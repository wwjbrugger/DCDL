"""
 train a dicision tree and to perform the sls algorithm with the inout and output_one_picture of a binary layer
"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import helper_methods as help
import numpy as np
import SLS_Algorithm as SLS
import pickle

def sls_convolution ( number_of_disjunction_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, one_against_all) :

    # load data for sls approximation
    label_set = np.load('data/label_SLS.npy')
    training_set = np.load('data/data_set_train.npy')
    shape = training_set.shape
    result_conv = np.load('data/result_conv.npy')
    kernel = np.load('data/kernel.npy')

    # add colour channel to data
    training_set=training_set.reshape((shape[0],shape[1],shape[2],1))
    # cast to True and False values
    training_set = help.transform_to_boolean(training_set)
    label_set = help.transform_to_boolean(label_set)
    kernel_width = kernel.shape[0]
    values_under_kernel = help.data_in_kernel(training_set, stepsize=stride_of_convolution, width=kernel_width)

    kernel_approximation = []

    for channel in range(label_set.shape[3]):
        # Update possibility (was not changed to be consistent with existing experiment results):
        # change to "Rule extraction for kernel {} "
        print("Ruleextraction for Kernel {} ".format(channel))
        training_set_flat, label_set_flat = help.permutate_and_flaten(values_under_kernel, label_set,
                                                                      channel_label=channel)
        # get formula for SLS blackbox approach
        found_formula = \
            SLS.rule_extraction_with_sls_without_validation(training_set_flat, label_set_flat,  number_of_disjunction_term_in_SLS,
                                                            Maximum_Steps_in_SKS, np.sign(np.reshape(kernel[:,:,:,channel], -1)))
        kernel_approximation.append(found_formula)
        # Update possibility (was not changed to be consistent with existing experiment results):
        # delete comment
        #label_self_calculated = help.calculate_convolution(values_under_kernel, kernel[:, :, :, channel], result_conv)
    # save formula as boolean_formel object
    pickle.dump(kernel_approximation, open('data/logic_formula_label_{}'.format(one_against_all), "wb"))

    #save formula in array_code as numpy array
    # Update possibility (was not changed to be consistent with existing experiment results):
    # change kernel_approximation = np.load('data/kernel_approximation_label_{}.npy'.format(one_against_all)) in reduce kernel to
    # logic_formula = pickle.load(open('data/logic_formula_label_{}'.format(one_against_all), "rb"))
    # kernel_approximation = np.reshape(logic_formula.formel_in_arrays_code, (-1, kernel_width, kernel_width))
    # than delete following lines
    formel_in_array_code = []
    for formel in kernel_approximation:
        formel_in_array_code.append(np.reshape(formel.formel_in_arrays_code, (-1, kernel_width, kernel_width)))
    np.save('data/kernel_approximation_label_{}.npy'.format(one_against_all), formel_in_array_code)
    return 


def visualize_kernel(one_against_all):
    # Update possibility (was not changed to be consistent with existing experiment results):
    #   Visualisation of the kernel is started
    print('Visualistation of the Kernel is started ')
    kernel = np.load('data/kernel.npy')

    # Update possibility (was not changed to be consistent with existing experiment results):
    # delete following line
    label_for_pic = ['kernel {} '.format(i) for i in range(kernel.shape[3])]

    help.visualize_singel_kernel(kernel, 28, "Kernel for {} againt all".format(one_against_all), set_vmin_vmax = True)

# Update possibility (was not changed to be consistent with existing experiment results):
#     delete main method
if __name__ == '__main__':
    Number_of_disjuntion_term_in_SLS = 4
    Maximum_Steps_in_SKS = 100
    stride_of_convolution = 28
    one_against_all = 2

    sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution,one_against_all)


