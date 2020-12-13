import tensorflow as tf
import get_data as get_data


def extract_data(neural_net,input_neural_net, operations_in_DCDL, print_nodes_in_neural_net ):
    dic_output_neural_net_bool = {}
    with tf.Session() as sess:
        neural_net.saver.restore(sess, neural_net.folder_to_save)
        if print_nodes_in_neural_net:
            print('Operations in graph \n', )
            operations =  [n.name for n in sess.graph.as_graph_def().node]
            for operation in operations:
                print(5 * '\t', operation)

        for key, node_dic in operations_in_DCDL.items():
            node_name = node_dic['name']
            # get operation from graph
            operation = \
                sess.graph.get_operation_by_name(node_name)
            # get output for operation if feed with input_neural_net
            output_operation = sess.run(operation.outputs[0],
                     feed_dict={neural_net.Input_in_Graph: input_neural_net})
            dic_output_neural_net_bool[node_name] = get_data.transform_to_boolean(output_operation)
    return dic_output_neural_net_bool


def extract_data_single_node(neural_net,input_neural_net, label, node_name):
    with tf.Session() as sess:
        neural_net.saver.restore(sess, neural_net.folder_to_save)
        # get operation from graph
        operation = \
            sess.graph.get_operation_by_name(node_name)
        # get output for operation if feed with input_neural_net
        if label is None:
            output_operation = sess.run(operation.outputs[0],
                     feed_dict={neural_net.Input_in_Graph: input_neural_net,})
        else:
            output_operation = sess.run(operation.outputs[0],
                                        feed_dict={neural_net.Input_in_Graph: input_neural_net,
                                                   neural_net.True_Label: label})
    return output_operation