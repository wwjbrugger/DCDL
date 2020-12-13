import os

os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import helper_methods as helper_vis
import numpy as np
import SLS_Algorithm as SLS


def sls_convolution(training_set, label_set, kernel_to_approximate, number_of_disjunction_term_in_SLS,
                    maximum_steps_in_SLS, stride_of_convolution, init_with_kernel):
    # load data for sls approximation
    shape = training_set.shape

    # add colour channel to data
    training_set = training_set.reshape((shape[0], shape[1], shape[2], 1))
    # cast to True and False values
    training_set = helper_vis.transform_to_boolean(training_set)
    label_set = helper_vis.transform_to_boolean(label_set)
    kernel_width = kernel_to_approximate.shape[0]
    values_under_kernel = helper_vis.data_in_kernel(training_set,
                                                        stepsize=stride_of_convolution,
                                                        width=kernel_width)

    kernel_approximation = []

    for channel in range(label_set.shape[3]):
        print("Rule extraction for kernel {} ".format(channel))
        training_set_flat, label_set_flat = helper_vis.permutate_and_flaten(
            data=values_under_kernel,
            label=label_set,
            channel_label=channel)

        # get formula for SLS blackbox approach
        found_formula = \
            SLS.rule_extraction_with_sls_without_validation(data=training_set_flat,
                                                            label=label_set_flat,
                                                            number_of_disjunction_term=number_of_disjunction_term_in_SLS,
                                                            maximum_steps_in_SLS=maximum_steps_in_SLS,
                                                            kernel = init_with_kernel)
        kernel_approximation.append(found_formula)

    return kernel_approximation
