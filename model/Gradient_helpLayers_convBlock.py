"""
    -@tf.RegisterGradient("BetterSTE")
     def _pw_grad(op, x)

    -@tf.RegisterGradient("ClippedSTE")
     def _pw_grad(op, x)

    -binarize_STE(x, name="egal")

    -binarize_ClippedSTE

    -binarize_BetterSTE(x, name="egal"):

    -rule_dense_eff(X, k, dim, nr, activation=binarize_STE)

    -rule_conv_block((X, nr, pretrain, skip=False, n_filter=16, s_filter=5, pool_by_stride=False, activation=binarize_STE,
                    bn_before=False, bn_after=False, ind_scaling=False, pool_before=False, pool_after=False, avg_pool=False)

    input_exp(X, nr, activation=binarize_STE, ind_scaling=False)
        Layer to transform real valued pictures into binarized


"""
import tensorflow as tf
import numpy as np
import sys

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Gradient Estimator +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


@tf.RegisterGradient("BetterSTE")
def _pw_grad(op, x):
    g = tf.where(x < -1, -tf.ones_like(x), tf.where(x > 1, tf.ones_like(x),
                                                    tf.where(x > 0, 2 * x - 2 * tf.square(x), 2 * x + 2 * tf.square(x))))
    return g


@tf.RegisterGradient("ClippedSTE")
def _pw_grad(op, x):
    return tf.clip_by_value(tf.identity(x), -1, 1)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Layer ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def binarize_STE(x, name="egal"):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "Identity"}):
        signed = tf.sign(x)
        #print_signed = tf.print(signed, output_stream=sys.stderr, name=name, summarize=-1)
        return signed
        #identity = tf.identity(x)
        #return identity


def binarize_ClippedSTE(x, name="egal"):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "ClippedSTE"}):
        signed = tf.sign(x)
        print_signed = tf.print(signed, output_stream=sys.stderr, name=name, summarize=-1)
        return signed


def binarize_BetterSTE(x, name="egal"):
    """
    Clip and binarize tensor using the straight through estimator (STE) for the gradient.
    """
    g = tf.get_default_graph()
    with g.gradient_override_map({"Sign": "BetterSTE"}):
        signed = tf.sign(x)
        #print_signed = tf.Print(signed, [signed], name=name)
        return signed


def rule_dense_eff(X, k, dim, nr, activation=binarize_STE):
    # Update possibility (was not changed to be consistent with existing experiment results):
    # not used in experiment
    with tf.variable_scope("dcdl" + str(nr), reuse=False):
        # Get trainable weights
        w = tf.get_variable("weights", dtype=tf.float32, initializer=tf.random_normal([dim, k], 0, 1),
                            trainable=True)
        b = tf.get_variable("bias", dtype=tf.float32, initializer=tf.random_normal([k], 0, 1),
                            trainable=True)

        reduced_acts = tf.matmul(X, w) + b
        thresholded = activation(reduced_acts)  # - tf.nn.relu(thresholds))) #- thresholds)

    return thresholded


def rule_conv_block(X, nr, pretrain, skip=False, n_filter=16, s_filter=5, stride = 2, pool_by_stride=False,
                    activation=binarize_STE,
                    bn_before=False, bn_after=False,
                    ind_scaling=False, pool_before=False,
                    pool_after=False, avg_pool=False):
    # Update possibility (was not changed to be consistent with existing experiment results):
    # not used in experiment
    with tf.variable_scope("dcdl_conv_" + str(nr), reuse=False):

        if skip:
            X_skip = tf.identity(X)
        else:
            X_skip = None

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Layer 1

        X = tf.layers.conv2d(inputs=X, filters=n_filter, kernel_size=[s_filter, s_filter], strides=[
                             stride, stride], padding="same", activation=activation, use_bias=False)

        if bn_before:
            X = tf.layers.batch_normalization(X, training=True)

        #X = activation(X, name="_out1")
        #X = tf.sign(X,name= 'Sign_operation_after_first_conv2d')
        if bn_after:
            X = tf.layers.batch_normalization(X, training=True)

        if ind_scaling:
            alpha = tf.get_variable("alpha1", dtype=tf.float32, initializer=tf.constant(
                1, dtype=tf.float32, shape=X.shape[1:]), trainable=True)
            X = X * alpha

        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # Layer 2
        stride = 2 if pool_by_stride else 1
        X = tf.layers.conv2d(inputs=X, filters=n_filter, kernel_size=[s_filter, s_filter], strides=[
                             stride, stride], padding="same", activation=None)

        if pool_by_stride and X_skip is not None:
            X_skip = tf.layers.max_pooling2d(X_skip, 2, 2, padding='same')

        if X_skip is not None:
            if X_skip.shape[3] != X.shape[3]:
                X_skip = tf.concat([X_skip, tf.zeros_like(X_skip)], 3)
            X = X + X_skip

        X = tf.sign(X, name='Sign_operation_after_secound_conv2d')

        if pool_before:
            if avg_pool:
                X = tf.layers.average_pooling2d(X, 2, 2, padding='same')
            else:
                X = tf.layers.max_pooling2d(X, 2, 2, padding='same')

        if bn_before:
            X = tf.layers.batch_normalization(X, training=True)

        activation(X, name="_out2")

        if bn_after:
            X = tf.layers.batch_normalization(X, training=True)

        if pool_after:
            if avg_pool:
                X = tf.layers.average_pooling2d(X, 2, 2, padding='same')
            else:
                X = tf.layers.max_pooling2d(X, 2, 2, padding='same')

        if ind_scaling:
            alpha = tf.get_variable("alpha2", dtype=tf.float32, initializer=tf.constant(
                1, dtype=tf.float32, shape=X.shape[1:]), trainable=True)
            X = X * alpha

    return X


def input_exp(X, nr, activation=binarize_STE, ind_scaling=False, name = "input"):
    # Update possibility (was not changed to be consistent with existing experiment results):
    # not used in experiment
    with tf.variable_scope(name + str(nr), reuse=False):
        b = tf.get_variable("bias", dtype=tf.float32, initializer=tf.random_normal(X.shape[1:], 0, 1),
                            trainable=True)
        alpha = tf.get_variable("alpha", dtype=tf.float32, initializer=tf.constant(
            1, dtype=tf.float32, shape=X.shape[1:]), trainable=True)

        thresholded = activation(X - b)
        if ind_scaling:
            thresholded = thresholded * alpha

    return thresholded

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# not used +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

