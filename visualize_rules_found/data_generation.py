import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

import model.visualize_rules_found.one_conv_block_model as model_one_convolution





def data_generation (network):

    train_nn = np.load('data/data_set_train.npy')
    label_train_nn = np.load('data/data_set_label_train_nn.npy')

    with tf.Session() as sess:
        # load trained net
        network.saver.restore(sess, network.folder_to_save)
        # code to get all tensors and nodes in the graph used. helpful to get the right names
        # tensors = [n.name for n in sess.graph.as_graph_def().node]
        # op = restored.sess.graph.get_operations()

        # data as the get into the net
        input = sess.graph.get_operation_by_name("Placeholder").outputs[0]

        # after reshaping
        operation_data_for_SLS = sess.graph.get_operation_by_name('Reshape')
        # data at the end of the first convolution block used as label to train DCDL
        operation_label_SLS = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Sign')
        # data after first convolution operation not needed in experiment
        operation_result_conv = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Conv2D')
        # kernel which is used in first convolution
        operation_kernel_conv_1_conv2d = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/kernel/read')
        # kernel which is used in dense layer
        operation_dense_kernel_read = sess.graph.get_operation_by_name('dense/kernel/read')

        # extract the (intermediate) results from the net
        input_for_SLS = sess.run(operation_data_for_SLS.outputs[0],
                                          feed_dict={input: train_nn})

        label_SLS = sess.run(operation_label_SLS.outputs[0],
                                          feed_dict={input: train_nn})

        result_conv = sess.run(operation_result_conv.outputs[0],
                                         feed_dict={input: train_nn})

        kernel_conv_1_conv2d = sess.run(operation_kernel_conv_1_conv2d.outputs[0],
                                         feed_dict={input: train_nn})
        dense_kernel_read = sess.run(operation_dense_kernel_read.outputs[0],
                                        feed_dict={input: train_nn})



        # save the (intermediate) results from the net
        np.save('data/label_SLS.npy', label_SLS)
        np.save('data/result_conv.npy', result_conv)
        np.save('data/kernel.npy', kernel_conv_1_conv2d)

        print('data generation is finished')

