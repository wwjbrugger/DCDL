import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import numpy as np
import SLS_Algorithm as SLS
import pickle
import helper_methods as helper_methods


def sls_convolution (number_of_disjuntion_term_in_SLS_DCDL, maximum_steps_in_SLS_DCDL, stride_of_convolution,
                     data_sign, label_sign, used_kernel, result, path_to_store , DCDL_train, unique_index) :
    # prepare data for learning/using DCDL rules to approximate convolutional layer
    kernel_width = used_kernel.shape[0]

    # get subsamples of data as they would be under the kernel in a convolution operation
    data_flat, label = helper_methods.prepare_data_for_sls(data_sign, label_sign, kernel_width, stride_of_convolution)
    np.save(path_to_store + '_data_flat.npy', data_flat)

    # list to collect n many formulas if n kernels are used.
    logic_formulas = []
    # save preprocessed data for prediction
    print('Shape of flatten data: ', data_flat.shape )
    if DCDL_train:
        # DCDL rules are learned. This part is skipped if DCDL is used in prediction mode
        # make input data unique, speeds up computation in experiment the performance was not harmed through this step
        if unique_index:
            _, unique_index = np.unique(data_flat, return_index=True, axis=0)
        data_flat = data_flat[unique_index]
        print('Shape of flatten data after making unique', data_flat.shape )
        for channel in range(label.shape[3]):
            # iterate through all channels of the label. For every channel a logical rule is learned
            print("ule extraction for kernel_conv {}".format(channel))
            #label_flat = label[:, :, :, channel].reshape(data_flat.shape[0])
            label_flat = label[:, :, :, channel].reshape(-1)[unique_index]

            # get rule for approximating the convolution
            found_formula = \
                SLS.rule_extraction_with_sls_without_validation(data=data_flat,
                                                                label=label_flat,
                                                                number_of_disjunction_term=number_of_disjuntion_term_in_SLS_DCDL,
                                                                maximum_steps_in_SLS=maximum_steps_in_SLS_DCDL)
            found_formula.shape_input_data = data_sign.shape
            found_formula.shape_output_data = label.shape
            logic_formulas.append(found_formula)

            accuracy = (data_flat.shape[0] - found_formula.total_error) / data_flat.shape[0]
            print("Accuracy of SLS: ", accuracy, "\n")

            if result is not None:
                # calculate manuel the convolution done in the neural net
                # for debugging not used in experiment.
                label_self_calculated = helper_methods.calculate_convolution(data_flat, used_kernel[:, :, :, channel], result)


        if path_to_store is not None:
            # store all found rules for making prediction later
            pickle.dump(logic_formulas, open(path_to_store, "wb"))
    return logic_formulas


def sls_dense(number_of_disjuntion_term_in_SLS_DCDL, maximum_steps_in_SLS_DCDL, data, label, path_to_store,
              SLS_Training):
    # control code for approximating the dense layer in the NN with DCDL
    data = helper_methods.transform_to_boolean(data)
    # bring data with channels into a flat form
    data_flat = np.reshape(data, (data.shape[0], -1))
    # path_to_store is path_to_use['logic_rules_dense']
    np.save(path_to_store + '_data_flat.npy', data_flat)
    print('Shape of flatten data: ', data_flat.shape)
    if SLS_Training:

        found_formula = \
            SLS.rule_extraction_with_sls_without_validation(data=data_flat,
                                                            label=label,
                                                            number_of_disjunction_term=number_of_disjuntion_term_in_SLS_DCDL,
                                                            maximum_steps_in_SLS=maximum_steps_in_SLS_DCDL)

        found_formula.shape_input_data = data.shape
        found_formula.shape_output_data = label.shape

        # calculate accuracy on train data
        accuracy = (data_flat.shape[0] - found_formula.total_error) / data_flat.shape[0]
        print("Accuracy of DCDL on train data: ", accuracy, '\n')

        if path_to_store is not None:
            pickle.dump(found_formula, open(path_to_store, "wb"))

