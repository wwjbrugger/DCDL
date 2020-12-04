import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf






def acc_data_generation ( network, path_to_use):
    """
    SLS_Training if False test_data are used in stead of training_data
    """
    print( 'Used data as input in graph: ', path_to_use['input_graph'])
    train_nn = np.load(path_to_use['input_graph'])

    if train_nn.ndim == 3:
        train_nn = train_nn.reshape((-1, 28, 28, 1))


    with tf.Session() as sess:
        network.saver.restore(sess, network.folder_to_save)
        # tensors = [n.name for n in sess.graph.as_graph_def().node]
        # op = restored.sess.graph.get_operations()

        input = sess.graph.get_operation_by_name("Placeholder").outputs[0]

        operation_reshape = sess.graph.get_operation_by_name('Reshape')
        
        operation_sign_con_1 = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Sign')
        operation_result_conv_1 = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Conv2D')
        operation_kernel_conv_1 = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/kernel/read')

        operation_max_pool_1 = sess.graph.get_operation_by_name('MaxPool')

        operation_sign_con_2 = sess.graph.get_operation_by_name('dcdl_conv_2/conv2d/Sign')
        operation_result_conv_2 = sess.graph.get_operation_by_name('dcdl_conv_2/conv2d/Conv2D')
        operation_kernel_conv_2 = sess.graph.get_operation_by_name('dcdl_conv_2/conv2d/kernel/read')

        operation_arg_max = sess.graph.get_operation_by_name('ArgMax')

        

        reshape = sess.run(operation_reshape.outputs[0],
                                          feed_dict={input: train_nn})

        sign_con_1 = sess.run(operation_sign_con_1.outputs[0],
                                          feed_dict={input: train_nn})

        result_conv_1 = sess.run(operation_result_conv_1.outputs[0],
                                         feed_dict={input: train_nn})

        kernel_conv_1= sess.run(operation_kernel_conv_1.outputs[0],
                                         feed_dict={input: train_nn})

        max_pool_1= sess.run(operation_max_pool_1.outputs[0],
                                         feed_dict={input: train_nn})
        

        sign_con_2 = sess.run(operation_sign_con_2.outputs[0],
                                          feed_dict={input: train_nn})

        result_conv_2 = sess.run(operation_result_conv_2.outputs[0],
                                         feed_dict={input: train_nn})

        kernel_conv_2= sess.run(operation_kernel_conv_2.outputs[0],
                                         feed_dict={input: train_nn})
        arg_max = sess.run( operation_arg_max.outputs[0],
                                         feed_dict={input: train_nn})



        np.save(path_to_use['g_reshape'], reshape)
        np.save(path_to_use['g_sign_con_1'], sign_con_1)
        np.save(path_to_use['g_result_conv_1'], result_conv_1)
        np.save(path_to_use['g_kernel_conv_1'], kernel_conv_1)

        np.save(path_to_use['g_max_pool_1'], max_pool_1)

        np.save(path_to_use['g_sign_con_2'], sign_con_2)
        np.save(path_to_use['g_result_conv_2'], result_conv_2)
        np.save(path_to_use['g_kernel_conv_2'], kernel_conv_2)
        np.save(path_to_use['g_arg_max'], arg_max)

        print('data generation is finished')



