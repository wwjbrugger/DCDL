import numpy as np

import visualize_rules_found.train_network as first
import visualize_rules_found.data_generation as secound
import visualize_rules_found.extract_logic_rules as third
import visualize_rules_found.reduce_kernel as fourths
import visualize_rules_found.helper_methods as help_vis

if __name__ == '__main__':
    # Update possibility (was not changed to be consistent with existing experiment results):
    #
    # data_set_to_use = 'cifar'  # 'numbers' or 'fashion'

    # length of one-hot-encoded label e.g.[0,1]
    number_classes_to_predict = 2

    # Update possibility (was not changed to be consistent with existing experiment results):
    #delete comment
    #dithering_used= 'floyd-steinberg'

    # If pictures should be dithered set values to 0 or 1 (see https://en.wikipedia.org/wiki/Dither)
    dithering_used = True
    # Label you want to test against all
    one_against_all_array =[0,1,2,3,4,5,6,7,8,9]

    # Update possibility (was not changed to be consistent with existing experiment results):
    #    maximum_steps_in_SLS
    Maximum_Steps_in_SKS = 2000
    # stride use in convolution
    stride_of_convolution = 28
    # Update possibility (was not changed to be consistent with existing experiment results):
    #    delete repetitions_of_sls
    repetitions_of_sls = 1
    # shape of kernel used in convolution
    shape_of_kernel = (28,28)
    # get the neural net for datasets with one colour channel
    network = first.model_one_convolution.network_one_convolution(shape_of_kernel=shape_of_kernel,
                                                                  nr_training_itaration=200,
                                                                  stride=stride_of_convolution, check_every=10,
                                                                  number_of_kernel=1,
                                                                  number_classes=number_classes_to_predict)
    # number of disjunctions to try
    # Update possibility (was not changed to be consistent with existing experiment results):
    #    k_interval
    K_interval = [150]
    for one_against_all in one_against_all_array:
        first.train_model(network, dithering_used, one_against_all, number_classes_to_predict = number_classes_to_predict)
        # writes (intermediate) results to files, which are used for training and evaluation DCDL
        secound.data_generation(network)
        # visualize kernel of nn
        third.visualize_kernel(one_against_all)

        # Update possibility (was not changed to be consistent with existing experiment results):
        #    replace Number_of_disjuntion_term_in_SLS with  number_of_disjunction_term_in_SLS
        for Number_of_disjuntion_term_in_SLS in K_interval:
            # Update possibility (was not changed to be consistent with existing experiment results):
            # delete result_of_reduction
            result_of_reduction = []
            # Update possibility (was not changed to be consistent with existing experiment results):
            # for loop not necessarily
            for i in range(repetitions_of_sls):
                # extract logical formula with SLS algorithm
                third.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, one_against_all)
                # Update possibility (was not changed to be consistent with existing experiment results):
                # write in two lines
                #  normalize logical rule for visualization
                result_of_reduction.append(fourths.reduce_SLS_results_of_one_run(one_against_all))
                """For only one run of SLS"""
                if repetitions_of_sls == 1:
                    help_vis.visualize_singel_kernel(np.reshape(result_of_reduction, -1), 28,
                                             'k= {}'.format(Number_of_disjuntion_term_in_SLS), set_vmin_vmax=True)



