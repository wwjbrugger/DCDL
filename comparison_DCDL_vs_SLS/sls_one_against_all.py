"""Script to run SLS with the input data of the neural network and the true label of this data"""
import os
os.environ["BLD_PATH"] = "../parallel_sls/bld/Parallel_SLS_shared"
import numpy as np
import SLS_Algorithm as SLS
import helper_methods as helper_methods
import pickle

import comparison_DCDL_vs_SLS.acc_data_generation as secound

def SLS_black_box_train(path_to_use, number_of_disjunction_term_in_SLS_BB, maximum_steps_in_SLS_BB, one_against_all):

    print('\n\n \t\t sls run ')
    print('number_of_disjunction_term_in_SLS_BB: ', number_of_disjunction_term_in_SLS_BB)

    # train set is loaded
    print('Input to train SLS black box is ',path_to_use['input_graph'])
    # prepare data for SLS
    training_set = np.load(path_to_use['input_graph'])
    training_set = helper_methods.transform_to_boolean(training_set)
    # flatten input from convolution or picture
    training_set_flat = np.reshape(training_set, (training_set.shape[0],-1))

    # load prediction NN or train data
    print('Label to train SLS black box is ',path_to_use['label_dense'])
    label_set = np.load(path_to_use['label_dense'])
    # prepare label for SLS
    if label_set.ndim == 1:
        # label are the prediction of the nn
        pass
    if label_set.ndim == 2:
        # if label are from true data
        # get the second position of the one-hot-label
        # if data belongs to  label 'one' then a 0 is writen out
        # if data belongs to  label rest then a 1 is writen out
        # in comparison to prediction of the nn this is necessarily because the prediction of the neural net
        # is an arg_max operation if the data belong to class one it will return 0
        # if the data belong to class rest it will return 1
        #label_set = [label[1] for label in label_set]
        label_set = [ np.argmax(one_hot_label) for one_hot_label in label_set]
    label_set = helper_methods.transform_to_boolean(label_set)
    label_set_flat = label_set

    # get formula for SLS blackbox approach
    found_formula=SLS.rule_extraction_with_sls(
        train=training_set_flat,
        train_label=label_set_flat,
        number_of_disjunction_term=number_of_disjunction_term_in_SLS_BB,
        maximum_steps_in_SLS=maximum_steps_in_SLS_BB,
        kernel=False,
        p_g1=.5,
        p_g2=.5,
        p_s=.5,
        batch=True,
        cold_restart=True,
        decay=0,
        min_prob=0,
        zero_init=False
    )

    # calculate accuracy on train set
    accuracy = (training_set.shape[0] - found_formula.total_error_on_validation_set) / training_set.shape[0]
    print("Accuracy of SLS: ", accuracy, '\n')
    # save formula for SLS blackbox approach
    pickle.dump(found_formula, open(path_to_use['logic_rules_SLS'], "wb"))
    if 'cifar' not in path_to_use['logs']:
        # visualize formula
        formula_in_arrays_code = np.reshape(found_formula.formula_in_arrays_code, (-1, 28, 28))
        reduced_kernel = helper_methods.reduce_kernel(formula_in_arrays_code, mode='norm')
        helper_methods.visualize_singel_kernel(kernel = np.reshape(reduced_kernel, (-1)),
                             kernel_width = 28,
                             title ='norm of all SLS Formula for {} against all \n  k= {}'.format(one_against_all,
                                                                                                          number_of_disjunction_term_in_SLS_BB))
    return found_formula, accuracy

def black_box_predicition (found_formula, path_to_use):
    # calculate prediction for SLS blackbox approach on test data
        print('Prediction with extracted rules from SLS for test data')
        print('Input data :', path_to_use['test_data'])
        print('Label :', path_to_use['test_label'])

        # load test data and prepare them
        test_data = np.load(path_to_use['test_data'])
        test_data_flat = np.reshape(test_data, (test_data.shape[0],-1))
        test_data_flat = helper_methods.transform_to_boolean(test_data_flat)

        # get the second position of the one-hot-label
        # if data belongs to  label 'one' then a 0 is writen out
        # if data belongs to  label rest then a 1 is writen out

        test_label = np.load(path_to_use['test_label'])
        #test_label = [label[1] for label in test_label]
        test_label = [np.argmax(one_hot_label) for one_hot_label in test_label]
        test_label = helper_methods.transform_to_boolean(test_label)

        path_to_store_prediction = path_to_use['logic_rules_SLS']
        # return accuracy  compared with test data
        return helper_methods.prediction_SLS_fast(test_data_flat, test_label, found_formula, path_to_store_prediction)


if __name__ == '__main__':
    # useful if only SLS-Blackbox should be used
    import accurancy_test.acc_main as main
    use_label_predicted_from_nn = True
    Input_from_SLS = None
    Training_set = True
    data_set_to_use = 'cifar'
    path_to_use = main.get_paths(Input_from_SLS, use_label_predicted_from_nn, Training_set, data_set_to_use)
    _, _, _, network = main.get_network(data_set_to_use, path_to_use)
    secound.acc_data_generation(network, path_to_use)

    found_formel = SLS_black_box_train(path_to_use)
    if not use_label_predicted_from_nn:
        black_box_predicition(found_formel)