import numpy as np

import visualize_rules_found.train_network as first
import visualize_rules_found.data_generation as secound
import visualize_rules_found.extract_logic_rules as third
import visualize_rules_found.reduce_kernel as fourths
import visualize_rules_found.helper_methods as help_vis

if __name__ == '__main__':
    number_classes_to_predict = 2


    #dithering_used= 'floyd-steinberg'
    dithering_used = True
    one_against_all_array =[0,1,2,3,4,5,6,7,8,9]

    Maximum_Steps_in_SKS = 2000
    stride_of_convolution = 28
    repetitions_of_sls = 1
    shape_of_kernel = (28,28)
    values_max_1 = True
    network = first.model_one_convolution.network_one_convolution(shape_of_kernel=shape_of_kernel,
                                                                  nr_training_itaration=200,
                                                                  stride=stride_of_convolution, check_every=10,
                                                                  number_of_kernel=1,
                                                                  number_classes=number_classes_to_predict)
    K_interval = [150]
    for one_against_all in one_against_all_array:
        first.train_model(network, dithering_used, one_against_all, number_classes_to_predict = number_classes_to_predict)

        secound.data_generation(network)

        third.visualize_kernel(one_against_all)

        for Number_of_disjuntion_term_in_SLS in K_interval:
            result_of_reduction = []
            for i in range(repetitions_of_sls):
                third.sls_convolution(Number_of_disjuntion_term_in_SLS, Maximum_Steps_in_SKS, stride_of_convolution, one_against_all)
                result_of_reduction.append(fourths.reduce_SLS_results_of_one_run(one_against_all))
                """For only one run of SLS"""
                if repetitions_of_sls == 1:
                    help_vis.visualize_singel_kernel(np.reshape(result_of_reduction, -1), 28,
                                             'k= {}'.format(Number_of_disjuntion_term_in_SLS), set_vmin_vmax=True)



