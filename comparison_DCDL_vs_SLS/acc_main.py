import model.two_conv_block_model as model_two_convolution
import comparison_DCDL_vs_SLS.acc_train as first
import comparison_DCDL_vs_SLS.acc_data_generation as second
import comparison_DCDL_vs_SLS.acc_extracting_pictures as third
import sys
import pandas as pd
import comparison_DCDL_vs_SLS.sls_one_against_all as sls
import time



def get_paths(input_from_SLS, use_label_predicted_from_nn, training_set, data_set_to_use):
    path_to_use = {
        # where the logs of the  neural net are saved
        'logs': 'data/{}/logs/'.format(data_set_to_use),
        # where the trained model is stored
        'store_model': 'data/{}/stored_models/'.format(data_set_to_use),
        # where the pandas frames are stored with the results for the comparison
        'results': 'data/{}/results/'.format(data_set_to_use),
        # where the data with which the neural net is trained are stored
        'train_data': 'data/{}/train_data.npy'.format(data_set_to_use),
        # where the label with which the neural net is trained are stored
        'train_label': 'data/{}/train_label.npy'.format(data_set_to_use),
        # where the validation data with which the neural net is chosen are stored
        'val_data': 'data/{}/val_data.npy'.format(data_set_to_use),
        # where the validation label with which the neural net is chosen are stored
        'val_label': 'data/{}/val_label.npy'.format(data_set_to_use),
        # where the data with which the neural net is evaluated are stored
        'test_data': 'data/{}/test_data.npy'.format(data_set_to_use),
        # where the data label which the neural net is evaluated are stored
        'test_label': 'data/{}/test_label.npy'.format(data_set_to_use),

        # Paths where the results of operations in the nn are stored
        # where the output of the node where the input data are flatten is stored
        'g_reshape': 'data/{}/data_reshape.npy'.format(data_set_to_use),
        # where the output of the sign operation node of the first convolution is stored
        'g_sign_con_1': 'data/{}/sign_con_1.npy'.format(data_set_to_use),
        #  where the output of the first convolution is stored (before sign_operation)
        'g_result_conv_1': 'data/{}/result_conv_1.npy'.format(data_set_to_use),
        # where the kernel/s of the first convolution is stored
        'g_kernel_conv_1': 'data/{}/kernel_conv_1.npy'.format(data_set_to_use),
        # where the output of the max_pooling node after the first convolution is stored
        'g_max_pool_1': 'data/{}/max_pool_1.npy'.format(data_set_to_use),
        # where the output of the sign operation node of the second convolution is stored
        'g_sign_con_2': 'data/{}/sign_con_2.npy'.format(data_set_to_use),
        # where the output of the second convolution is stored (before sign_operation)
        'g_result_conv_2': 'data/{}/result_conv_2.npy'.format(data_set_to_use),
        # where the kernel/s of the second convolution is stored
        'g_kernel_conv_2': 'data/{}/kernel_conv_2.npy'.format(data_set_to_use),
        # where the prediction of the neural net is stored /s of the second convolution is stored
        'g_arg_max': 'data/{}/arg_max.npy'.format(data_set_to_use),

        #Paths where the necessarily data to aproximate the neural net with logic formula are stored
        # where to store the logic formulas to approximate the first convolution
        'logic_rules_conv_1': 'data/{}/logic_rules_Conv_1'.format(data_set_to_use),
        # where to store the flatten data which are the input of the the first convolution approximation
        'flat_data_conv_1': 'data/{}/logic_rules_Conv_1_data_flat.npy'.format(data_set_to_use),
        # where to store the prediction of the the first convolution approximation
        'prediction_conv_1': 'data/{}/prediction_for_conv_1.npy'.format(data_set_to_use),

        # where to store the logic formulas to approximate the second convolution
        'logic_rules_conv_2': 'data/{}/logic_rules_Conv_2'.format(data_set_to_use),
        # where to store the flatten data which are the input of the the second convolution approximation
        'flat_data_conv_2': 'data/{}/logic_rules_Conv_2_data_flat.npy'.format(data_set_to_use),
        # where to store the prediction of the the second convolution approximation
        'prediction_conv_2': 'data/{}/prediction_for_conv_2.npy'.format(data_set_to_use),

        # where to store the logic formulas to approximate the dense layer
        'logic_rules_dense': 'data/{}/logic_rules_dense'.format(data_set_to_use),
        # where to store the flatten data which are the input of the the dense layer approximation
        'flat_data_dense': 'data/{}/logic_rules_dense_data_flat.npy'.format(data_set_to_use),
        # where to store the prediction of the the dense layer approximation
        'prediction_dense': 'data/{}/prediction_dense.npy'.format(data_set_to_use),

        # where to store the logic formulas of the blackbox approximation of the neural net
        'logic_rules_SLS': 'data/{}/logic_rules_SLS'.format(data_set_to_use),
    }
    # use input of the neural net to train the first approximation
    path_to_use['input_conv_1'] = path_to_use['g_reshape']
    # use output of the neural net to train the approximations
    path_to_use['label_conv_1'] = path_to_use['g_sign_con_1']
    path_to_use['label_conv_2'] = path_to_use['g_sign_con_2']

    # for comparing if the SLS blackbox approach or the DCDL approach has a higher similarity to the prediction
    # of the neural net with unknown data, we have to fill the graph with test data
    if training_set:
        # Graph is filled with the train data
        path_to_use['input_graph'] = path_to_use['train_data']
    else:
        # Graph is filled with the test data
        path_to_use['input_graph'] = path_to_use['test_data']

    if input_from_SLS:
        # prediction of the previous approximation is used for training the following approximation
        # default mode
        path_to_use['input_conv_2'] = 'data/{}/prediction_for_conv_1.npy'.format(data_set_to_use)
        path_to_use['input_dense'] = 'data/{}/prediction_for_conv_2.npy'.format(data_set_to_use)
    else:
        # results of the neural net is used for training the following approximation
        path_to_use['input_conv_2'] = path_to_use['g_max_pool_1']
        path_to_use['input_dense'] = 'data/{}/sign_con_2.npy'.format(data_set_to_use)

    if use_label_predicted_from_nn and training_set:
        # SLS black box is using prediction of neural net for training
        # DCDL is using prediction of neural net for training
        path_to_use['label_dense'] = path_to_use['g_arg_max']
    elif use_label_predicted_from_nn and not training_set:
        # SLS black box similarity on test data compared with prediction of neural net is calculated
        path_to_use['label_dense'] = path_to_use['g_arg_max']

    elif not use_label_predicted_from_nn and training_set:
        # SLS black box is using true label of data for training
        path_to_use['label_dense'] = path_to_use['train_label']
    elif not use_label_predicted_from_nn and not training_set:
        # SLS black box is using true data label to calculate accuracy
        # DCDL  is using true data label to calculate accuracy
        path_to_use['label_dense'] = path_to_use['test_label']

    return path_to_use


def get_network(data_set_to_use, path_to_use, convert_to_gray):
    # get network with two convolution and one dense layer at the end
    # net for dataset 'numbers' (MNIST) and 'fashion' (Fashion-MNIST)
    # have one colour channel as input writen in from [num_pic, pic_width, pic_height, num_colour_channel]
    # net for dataset 'cifar' (CIFAR)
    # have there colour channel as input writen in from [num_pic, pic_width, pic_height, num_colour_channel]

    # how many classes we want to distinguish, we are using one-against-all_testing so there are 2 classes
    number_classes_to_predict = 2
    # stride use in convolution
    stride_of_convolution = 2
    # shape of kernel used in convolution
    shape_of_kernel = (2, 2)
    # how many kernels are used in convolution. Every kernel is approximated by a logical formula.
    number_of_kernels = 8
    # name under which the model is stored after training
    name_of_model = '{}_two_conv_2x2'.format(data_set_to_use)

    if data_set_to_use in 'numbers' or data_set_to_use in 'fashion':
        network = model_two_convolution.network_two_convolution(path_to_use, name_of_model=name_of_model,
                                                                shape_of_kernel=shape_of_kernel,
                                                                nr_training_itaration=2000,
                                                                stride=stride_of_convolution,
                                                                number_of_kernel=number_of_kernels,
                                                                number_classes=number_classes_to_predict)
    elif data_set_to_use in 'cifar':

        if convert_to_gray:
            # dataset will be first converted to gray scale picture.
            # After this the dither operation is applied on the grey scale picture.
            # Not used in final run
            input_channels = 1
            input_shape = (None, 32, 32, 1)
        else:
            input_channels = 3
            input_shape = (None, 32, 32, 3)

        # get the neural net which will be trained later
        network = model_two_convolution.network_two_convolution(path_to_use, name_of_model=name_of_model,
                                                                shape_of_kernel=shape_of_kernel,
                                                                nr_training_itaration=2000,
                                                                stride=stride_of_convolution,
                                                                number_of_kernel=number_of_kernels,
                                                                number_classes=number_classes_to_predict,
                                                                input_channels=input_channels, input_shape=input_shape,
                                                                )

    return shape_of_kernel, stride_of_convolution, number_of_kernels, network

def get_pandas_frame (data_set_to_use, one_against_all):
    # returns an empty dataframe in which the results of the experiment will be saved.
    # has the form (X will be filed during experiment, None will not be filled, :

    # data_type | Used_label         | Concat | SLS prediction | SLS train | Neural network
    #  train    | Prediction_from_NN |   X           X            None           None
    #  train    | True_Label_of_Data |   X          None           X               X
    #  test     | Prediction_from_NN |   X           X            None           None
    #  test     | True_Label_of_Data |   X          None           X               X

    # the column data_type represent if training or test data where used
    # the Used_label column represents if the output of the Neural net (similarity measure)
    # or the true label (accuracy measure) was used for training/testing

    # Update possibility (was not changed to be consistent with existing experiment results):
    #   In the paper DCDL is used instead of Concat
    #   In the paper BB Prediction is used instead of SLS prediction
    #   In the paper BB Train is used instead of SLS train

    column_name = ['data_type', 'Used_label', 'DCDL', 'SLS BB prediction', 'SLS BB train', 'Neural network']
    row_index = [0, 1, 2, 3]
    df = pd.DataFrame(index=row_index, columns=column_name)
    df.at[0, 'data_type'] = 'train'
    df.at[1, 'data_type'] = 'train'
    df.at[2, 'data_type'] = 'test'
    df.at[3, 'data_type'] = 'test'
    df.at[0, 'Used_label'] = 'Prediction_from_NN'
    df.at[1, 'Used_label'] = 'True_Label_of_Data'
    df.at[2, 'Used_label'] = 'Prediction_from_NN'
    df.at[3, 'Used_label'] = 'True_Label_of_Data'

    # change setting of pandas frame to be shown without cutting in terminal
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', -1)
    return df

def fill_data_frame_with_DCDL(results, result_DCDL, use_label_predicted_from_nn, training_set):
    # fill cell in pandas frame with results of DCDL from previous loop.

    if use_label_predicted_from_nn and training_set:
        # similarity between DCDL and prediction of NN on train data
        results.at[0,'DCDL'] = result_DCDL
    if not use_label_predicted_from_nn and training_set:
        # accuracy  between DCDL and true label on train data
        results.at[1,'DCDL'] = result_DCDL
    if use_label_predicted_from_nn and not training_set:
        # similarity between DCDL and prediction of NN on test data
        results.at[2,'DCDL'] = result_DCDL
    if not use_label_predicted_from_nn and not training_set:
        # accuracy between DCDL and true label on test data
        results.at[3,'DCDL'] = result_DCDL


def fill_data_frame_with_sls_bb(results, result_SLS_BB_train, result_SLS_BB_test, use_label_predicted_from_nn):
    if use_label_predicted_from_nn:
        results.at[0, 'SLS BB prediction'] = result_SLS_BB_train
        results.at[3, 'SLS BB prediction'] = result_SLS_BB_test
    else:
        results.at[1, 'SLS BB train'] = result_SLS_BB_train
        results.at[3, 'SLS BB train'] = result_SLS_BB_test



if __name__ == '__main__':
    # start script with python acc_main to use parameter set in script
    # start script with python acc_main [dataset] [label for one against all]
    #   e.g. python acc_main fashion 6
    # will run the experiment for the Fashion Mnist Dataset with label 6 against all.

    if len(sys.argv) > 1:
    # start script with parameter
        print("used Dataset: ", sys.argv [1])
        print("Label-against-all", sys.argv [2])
        if (sys.argv[1] in 'numbers') or (sys.argv[1] in'fashion') or (sys.argv[1] in 'cifar'):
            data_set_to_use = sys.argv [1]
            one_against_all_l = [int(sys.argv [2])]
        else:
            raise ValueError('You choose a dataset which is not supported. \n Datasets which are allowed are numbers(Mnist), fashion(Fashion-Mnist) and cifar')
    else:
        # values if you start script without parameter
        data_set_to_use = 'cifar'  # 'numbers' or 'fashion'
        one_against_all_l = [4]

    for one_against_all in one_against_all_l:
        # timestr to make paths unique
        timestr = time.strftime("%Y%m%d-%H%M%S")
        # set size of train and validation data
        if data_set_to_use in 'cifar':
            size_train_nn = 45000
        else:
            size_train_nn = 55000
        size_valid_nn = 5000

        # If pictures should be dithered set values to 0 or 1 (see https://en.wikipedia.org/wiki/Dither)
        dithering_used = True
        # pictures in array have a scaled into range of [0 , 1]
        values_max_1 = True
        # covert colour picture into greyscale picture before dither
        convert_to_grey = False
        # if before calculating DCDL of convolution block go through input data and make data unique
        # speeds up calculation. If not set to through input into sls can be huge
        unique_index = True

        # The experiment is structured in 4 rounds.
        # first round:
        #   - neural net is trained
        #   - neural net accuracy is evaluated on test data
        #   - DCDL is trained with the prediction of neural net
        #   - DCDL is evaluated how good similarity to prediction on train data are
        #   - SLS-Blackbox-Prediction is trained with the prediction of the neural net
        #   - SLS-Blackbox-Prediction is evaluated how good similarity to prediction on train data are
        #   - SLS-Blackbox-Prediction rules are evaluated on true label of test data
        #  second round:
        #   - DCDL is evaluated  how good accuracy to true label of train data are
        #   - SLS-Blackbox-Trained is trained with true label of data
        #   - SLS-Blackbox-Trained is evaluated how good accuracy to true label of test are
        # third round:
        #   - DCDL is evaluate how good similarity to the prediction of the neural net with test data are
        # forth round:
        #   - DCDL is evaluate how good accuracy with the true label of test data are

        # If neural net should be trained
        NN_train_l = [True, False, False, False]
        # If DCDL should be trained
        # Update possibility (was not changed to be consistent with existing experiment results):
           # rename to DCDL_train_l
        DCDL_train_l = [True, False, False, False]
        # Use training set as input for this run ot the test set
        training_set_l = [True, True, False, False]
        # Use prediction of the neural net for run as label or true label
        use_label_predicted_from_nn_l = [True, False, True,
                                         False]
        # Use DCDL results of previous layer as Input in the approximation of current layer
        input_from_SLS = True
        # Output on terminal at the begin of each round to structure Output

        mode = ['Input data in net: train \t Label used for DCDL: Prediction of net',
                   'Input data in net: train \t Label used for DCDL: True label of data',
                   'Input data in net: test  \t Label used for DCDL: Prediction of net',
                   'Input data in net: test  \t Label used for DCDL: True label of data']

        # Set k for DCDL approximation
        number_of_disjuntion_term_in_SLS_DCDL = 40
        number_of_disjuntion_term_in_SLS_BB = 40
        # How many steps in SLS are made
        maximum_steps_in_SLS_DCDL = 2000
        maximum_steps_in_SLS_BB = 2500

        # get empty pandas frame to store results of experiment
        results = get_pandas_frame(data_set_to_use=data_set_to_use,
                                   one_against_all=one_against_all)

        for i in range(len(NN_train_l)):
            NN_train = NN_train_l[i]
            DCDL_train = DCDL_train_l[i]
            training_set = training_set_l[i]
            use_label_predicted_from_nn = use_label_predicted_from_nn_l[i]

            # the get paths methods control which data are used in a run
            path_to_use = get_paths(input_from_SLS=input_from_SLS,
                                    use_label_predicted_from_nn=use_label_predicted_from_nn,
                                    training_set=training_set,
                                    data_set_to_use=data_set_to_use)

            # get some variables concerning the net
            # Update possibility (was not changed to be consistent with existing experiment results):
            #    should be moved outside of the loop because it needs only to be called once.
            shape_of_kernel, stride_of_convolution, number_of_kernels, network = get_network(data_set_to_use=data_set_to_use,
                                                                                             path_to_use=path_to_use,
                                                                                             convert_to_gray=convert_to_grey)

            if NN_train:
                first.train_model(network = network ,
                                  dithering_used = dithering_used,
                                  one_against_all = one_against_all,
                                  data_set_to_use = data_set_to_use,
                                  path_to_use = path_to_use,
                                  convert_to_grey = convert_to_grey,
                                  results = results,
                                  size_train_nn = size_train_nn,
                                  size_valid_nn = size_valid_nn,
                                  values_max_1 = values_max_1)

            print('\n\n\n\t\t\t', mode[i])
            # writes (intermediate) results to files, which are used for training and evaluation DCDL
            second.acc_data_generation(network=network, path_to_use=path_to_use)
            # Update possibility (was not changed to be consistent with existing experiment results):
            # delete the following line
            # third.visualize_kernel(one_against_all, 'data/kernel_conv_1.npy')

            # If SLS Training is True DCDL formula for first convolution  is learned.
            # If its False the method make the necessarily preprocessing to
            # use the logic formulas and skip the training part
            third.DCDL_Conv_1(number_of_disjuntion_term_in_SLS_DCDL=number_of_disjuntion_term_in_SLS_DCDL,
                             maximum_steps_in_SLS_DCDL = maximum_steps_in_SLS_DCDL,
                             stride_of_convolution=stride_of_convolution,
                             DCDL_train=DCDL_train,
                             path_to_use=path_to_use,
                             unique_index=unique_index)

            # use learned DCDL formula for making prediction
            third.prediction_DCDL_1(path_to_use=path_to_use)

            # If SLS Training is True DCDL formula for second convolution is learned.
            # If its False the method make the necessarily preprocessing to
            # use the logic formulas and skip the training part
            third.SLS_DCDL_2(number_of_disjuntion_term_in_SLS_DCDL=number_of_disjuntion_term_in_SLS_DCDL,
                             maximum_steps_in_SLS_DCDL=maximum_steps_in_SLS_DCDL,
                             stride_of_convolution=stride_of_convolution,
                             DCDL_train=DCDL_train,
                             input_from_SLS = input_from_SLS,
                             path_to_use = path_to_use,
                             unique_index=unique_index)

            # use learned DCDL formula for making prediction
            third.prediction_DCDL_2(path_to_use=path_to_use)

            # If SLS Training is True DCDL formula for dense layer is learned.
            # If its False the method make the necessarily preprocessing to
            # use the logic formulas and skip the training part
            third.DCDL_dense(number_of_disjuntion_term_in_SLS_DCDL=number_of_disjuntion_term_in_SLS_DCDL,
                            maximum_steps_in_SLS_DCDL = maximum_steps_in_SLS_DCDL,
                            DCDL_train=DCDL_train,
                            path_to_use=path_to_use)

            # use learned DCDL formula for making prediction
            # returns an accuracy/similarity score compared to the label which is used in this run
            result_DCDL = third.prediction_dense(path_to_use=path_to_use)

            # fill result dataframe with the results of the DCDL approach
            fill_data_frame_with_DCDL(results =results,
                                      result_DCDL = result_DCDL,
                                      use_label_predicted_from_nn =use_label_predicted_from_nn ,
                                      training_set=training_set)

            # Methods to train and control SLS blackbox approaches
            if training_set:
                # First time trained with the prediction of the neural net fot the train set
                # Second time trained wit the true label of the train set

                # train SLS blackbox approach
                # found formula is the formula which is found by the SLS algorithm
                found_formula_SLS_BB, result_SLS_BB_train = sls.SLS_black_box_train(path_to_use,
                                                                          number_of_disjuntion_term_in_SLS_BB=number_of_disjuntion_term_in_SLS_BB,
                                                                          maximum_steps_in_SLS_BB=maximum_steps_in_SLS_BB,
                                                                          one_against_all=one_against_all          )
                # evaluate found formula always on test data
                result_SLS_BB_test = sls.black_box_predicition(found_formula=found_formula_SLS_BB,
                                                            path_to_use=path_to_use)
                # fill result dataframe with the results of the SLS blackbox approach
                fill_data_frame_with_sls_bb(results = results,
                                            result_SLS_BB_train=result_SLS_BB_train,
                                            result_SLS_BB_test=result_SLS_BB_test,
                                            use_label_predicted_from_nn=use_label_predicted_from_nn)



        # print and store result frame
        print(results, flush=True)

        print('path_to_store_results' , path_to_use['results']+ 'label_{}__{}'.format(one_against_all, timestr ))
        results.to_pickle(path_to_use['results']+ 'label_{}__{}'.format(one_against_all, timestr))
