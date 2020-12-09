import numpy as np
import tensorflow as tf


def binarize_STE(x):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "Identity"}):
        # actually returns three values -1, 0, 1 not only two as necessarily
        # signed = tf.sign(x)
        # tf.sign(x) returns `y = sign(x) = -1` if `x < 0`; 0 if `x == 0`; 1 if `x > 0`
        # this line returns
        # -1` if `x =< 0`;  1 if `x > 0
        signed = tf.sign(tf.sign(x) - 0.1)
        return signed


class network_two_convolution():
    def __init__(self,
                 path_to_store_model,
                 name_of_model,
                 learning_rate,
                 # length of one-hot-encoded label e.g.[0,1], after one_against_all
                 number_classes,
                 input_shape,
                 nr_training_iteration,
                 batch_size,
                 print_every,
                 check_every,
                 number_of_kernel_conv_1,
                 number_of_kernel_conv_2,
                 shape_of_kernel_conv_1,
                 shape_of_kernel_conv_2,
                 stride_conv_1,
                 stride_conv_2,
                 input_channels,
                 # activation is a sign function sign(x) = -1 if x <= 0, 1 if x > 0.
                 activation_str,
                 use_bias_in_conv_1,
                 use_bias_in_conv_2,
                 shape_max_pooling_layer,
                 stride_max_pooling,
                 dropout_rate,
                 # use arg_min function to cast one hot label to true or false
                 arg_min_label,
                 logging,
                 save_path_logs):

        # Clears the default graph stack and resets the global default graph.
        tf.compat.v1.reset_default_graph()
        # save method parameter as class parameter
        self.learning_rate = learning_rate
        self.classes = number_classes
        self.input_shape = input_shape
        self.nr_training_iteration = nr_training_iteration
        self.batch_size = batch_size
        self.print_every = print_every
        self.check_every = check_every
        self.folder_to_save = path_to_store_model
        self.name_of_model = name_of_model
        self.number_of_kernel_conv_1 = number_of_kernel_conv_1
        self.number_of_kernel_conv_2 = number_of_kernel_conv_2
        self.shape_of_kernel_conv_1 = shape_of_kernel_conv_1
        self.shape_of_kernel_conv_2 = shape_of_kernel_conv_2
        self.stride_conv_1 = stride_conv_1
        self.stride_conv_2 = stride_conv_2
        self.input_channels = input_channels
        self.shape_max_pooling_layer = shape_max_pooling_layer
        self.stride_max_pooling = stride_max_pooling
        self.dropout_rate = dropout_rate
        self.logging = logging
        self.save_path_logs = save_path_logs


        if activation_str in 'binarize_STE':
            self.activation = binarize_STE
        elif activation_str in 'relu':
            self.activation = tf.nn.relu

        self.use_bias_in_conv_1 = use_bias_in_conv_1
        self.use_bias_in_conv_2 = use_bias_in_conv_2

        self.arg_min_label = arg_min_label

        # built the graph which is used later
        self.built_graph()

    def built_graph(self):

        self.Input_in_Graph = tf.compat.v1.placeholder(
            dtype=tf.compat.v1.float32,
            shape=(None, self.input_shape[0], self.input_shape[1], self.input_channels))
        # True Label are one hot label with shape e.g. [[0,1], ... , [1,0]]
        self.True_Label = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[None, self.classes])
        X = self.Input_in_Graph
        with tf.compat.v1.variable_scope("dcdl_conv_1", reuse=False):
            # get first convolution block
            X = tf.compat.v1.layers.conv2d(inputs=X,
                                           filters=self.number_of_kernel_conv_1,
                                           kernel_size=self.shape_of_kernel_conv_1,
                                           strides=[self.stride_conv_1, self.stride_conv_1],
                                           padding="same",
                                           activation=self.activation,
                                           use_bias=self.use_bias_in_conv_1)

        X = tf.compat.v1.nn.max_pool(X, self.shape_max_pooling_layer, strides=self.stride_max_pooling, padding='SAME')

        with tf.compat.v1.variable_scope('dcdl_conv_2', reuse=False):
            # get second convolution block
            X = tf.compat.v1.layers.conv2d(inputs=X, filters=self.number_of_kernel_conv_2,
                                           kernel_size=self.shape_of_kernel_conv_2,
                                           strides=[self.stride_conv_2, self.stride_conv_2], padding="same",
                                           activation=self.activation,
                                           use_bias=self.use_bias_in_conv_2)

        X = tf.compat.v1.nn.dropout(X, rate=self.dropout_rate)
        X = tf.compat.v1.layers.flatten(X)

        # Update possibility (was not changed to be consistent with existing experiment results):
        # add properties
        # Computes softmax activations
        # inputs = X,
        # units = self.classes
        # activation = tf.compat.v1.nn.softmax
        self.prediction = tf.compat.v1.layers.dense(X, self.classes, tf.compat.v1.nn.softmax)

        # calculate loss
        self.loss = tf.compat.v1.reduce_mean(-tf.compat.v1.reduce_sum(self.True_Label *
                                                                      tf.compat.v1.log(self.prediction + 1E-10),
                                                                      reduction_indices=[1]))  # + reg2
        # make step with optimizer
        self.step = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Evaluate model
        # self.prediction has form [[p1,p2],[p1,p2], ...]
        # arg_max get index of higher value of the prediction
        if self.arg_min_label:
            # use argmin function to reduce one hot label to one number
            self.one_hot_out = tf.compat.v1.argmin(self.prediction, 1)
            self.hits = tf.compat.v1.equal(self.one_hot_out, tf.compat.v1.argmin(self.True_Label, 1))
        else:
            # use argmax function to reduce one hot label to one number
            self.one_hot_out = tf.compat.v1.argmax(self.prediction, 1)
            self.hits = tf.compat.v1.equal(self.one_hot_out, tf.compat.v1.argmax(self.True_Label, 1))

        self.accuracy = tf.compat.v1.reduce_mean(tf.compat.v1.cast(self.hits, tf.compat.v1.float32))

        # Initialize the variables
        self.init = tf.compat.v1.global_variables_initializer()

        # Save model

        self.saver = tf.compat.v1.train.Saver()

    def training(self, train, label_train, val, label_val, logging):
        loss_list, val_list = [], []
        with tf.compat.v1.Session() as sess:
            if logging:
                # logs can be visualized in tensorboard.
                # useful for see structure of the graph
                # Update possibility (was not changed to be consistent with existing experiment results):
                # delete following comment

                writer = tf.compat.v1.summary.FileWriter(self.save_path_logs, session=sess,
                                                         graph=sess.graph)  # + self.name_of_model, sess.graph)

            sess.run(self.init)
            best_acc_so_far = 0

            for iteration in range(self.nr_training_iteration):
                # train net
                indices = np.random.choice(len(train), self.batch_size)
                batch_X = train[indices]
                batch_Y = label_train[indices]
                feed_dict = {self.Input_in_Graph: batch_X, self.True_Label: batch_Y}
                # self step calls the optimizer
                _, lo, acc = sess.run([self.step, self.loss, self.accuracy], feed_dict=feed_dict)
                if iteration % self.print_every == 1:
                    print("Iteration: ", iteration, "Acc. at trainset: ", acc, flush=True)

                if iteration % self.check_every == 1:
                    # validate net on train data
                    # Update possibility (was not changed to be consistent with existing experiment results):
                    # 5000 should be a variable which is set in the main script
                    # e.g. size_of_val_used_in_one_step = 5000
                    indices = np.random.choice(len(val), 5000)
                    acc, lo = sess.run([self.accuracy, self.loss], feed_dict={
                        self.Input_in_Graph: val[indices], self.True_Label: label_val[indices]})
                    print("step: ", iteration, 'Accuracy at validation_set: ', acc, )

                    loss_list.append(lo)
                    val_list.append(acc)

                    if acc > best_acc_so_far:
                        # if accuracy at validation set is better than previous
                        best_acc_so_far = acc
                        save_path = self.saver.save(sess, self.folder_to_save)
                        print('Path to store parameter: ', save_path)

    def evaluate(self, input, label):
        # get accuracy for a data and label set
        with tf.compat.v1.Session() as sess:
            # load saved model variables
            self.saver.restore(sess, self.folder_to_save)
            acc = sess.run([self.accuracy], feed_dict={self.Input_in_Graph: input, self.True_Label: label})[0]
            # Update possibility (was not changed to be consistent with existing experiment results):
            # delete print statement
            print("Test Accuracy", self.name_of_model, acc)
            return acc
