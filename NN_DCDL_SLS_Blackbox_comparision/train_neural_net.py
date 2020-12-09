def create_new_net(general_settings_dic, setting_dic_NN):

        # get network with two convolution and one dense layer at the end
        # net for dataset 'numbers' (MNIST) and 'fashion' (Fashion-MNIST)
        # have one colour channel as input writen in from [num_pic, pic_width, pic_height, num_colour_channel]
        # net for dataset 'cifar' (CIFAR)
        # have there colour channel as input writen in from [num_pic, pic_width, pic_height, num_colour_channel]

        name_of_model = setting_dic_NN['name_of_model']
        if data_set_to_use in 'numbers' or data_set_to_use in 'fashion':
            # get the neural net for datasets with one colour channel
            network = model_two_convolution.network_two_convolution(path_to_store_model=path_to_store_model
                                                                    , name_of_model=name_of_model,
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

            # get the neural net for datasets with three colour channel
            network = model_two_convolution.network_two_convolution(path_to_store_model=path_to_store_model,
                                                                    name_of_model=name_of_model,
                                                                    shape_of_kernel=shape_of_kernel,
                                                                    nr_training_itaration=2000,
                                                                    stride=stride_of_convolution,
                                                                    number_of_kernel=number_of_kernels,
                                                                    number_classes=number_classes_to_predict,
                                                                    input_channels=input_channels,
                                                                    input_shape=input_shape,
                                                                    )

        return shape_of_kernel, stride_of_convolution, number_of_kernels, network

def train_neural_net():
    network.training(train=train_nn,
                     label_train=label_train_nn,
                     val=val,
                     label_val=label_val,
                     path_to_use=path_to_use)

    print("\n Start evaluate with train set ")
    # save accuracy on the NN on train set in results
    results.at[1, 'Neural network'] = network.evaluate(input=train_nn, label=label_train_nn)

    print("\n Start evaluate with validation set ")
    # should be the same value as the highest during training
    network.evaluate(input=val, label=label_val)

    print("\n Start evaluate with test set ")
    # save accuracy of the NN on test set in results
    results.at[3, 'Neural network'] = network.evaluate(input=test, label=label_test)

    print('end')