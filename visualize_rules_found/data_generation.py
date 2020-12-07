import warnings

warnings.filterwarnings('ignore', category=FutureWarning)
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf

import model.visualize_rules_found.one_conv_block_model as model_one_convolution





def data_generation (network, data_dic, data_type):

    input_dither = data_dic['dither_{}_data'.format(data_type)]
    label_train_nn = data_dic['{}_label_one_hot'.format(data_type)]

    with tf.Session() as sess:
        # load trained net
        network.saver.restore(sess, network.folder_to_save)
        # code to get all tensors and nodes in the graph used. helpful to get the right names
        # tensors = [n.name for n in sess.graph.as_graph_def().node]
        # op = restored.sess.graph.get_operations()

        # data as the get into the net
        input = sess.graph.get_operation_by_name("Placeholder").outputs[0]

        # data at the end of the first convolution block used as label to train DCDL
        operation_conv2d_Sign = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Sign')
        # data after first convolution operation not needed in experiment
        operation_result_Conv2D = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/Conv2D')
        # kernel which is used in first convolution
        operation_kernel_conv_1_conv2d = sess.graph.get_operation_by_name('dcdl_conv_1/conv2d/kernel/read')
        # kernel which is used in dense layer
        operation_dense_kernel_read = sess.graph.get_operation_by_name('dense/kernel/read')

        # extract the (intermediate) results from the net
        output_conv2d_Sign = sess.run(operation_conv2d_Sign.outputs[0],
                                          feed_dict={input: input_dither})

        result_Conv2D = sess.run(operation_result_Conv2D.outputs[0],
                                         feed_dict={input: input_dither})

        kernel_conv_1_conv2d = sess.run(operation_kernel_conv_1_conv2d.outputs[0],
                                         feed_dict={input: input_dither})
        dense_kernel_read = sess.run(operation_dense_kernel_read.outputs[0],
                                        feed_dict={input: input_dither})



        # save the (intermediate) results from the net
        data_dic['output_nn_dcdl_conv_1_conv2d_Sign_{}'.format(data_type)] = output_conv2d_Sign
        data_dic['output_nn_dcdl_conv_1_result_Conv2D_{}'.format(data_type)] =  result_Conv2D
        data_dic['output_nn_kernel_conv_1_conv2d'] = kernel_conv_1_conv2d
        print('data generation is finished')

