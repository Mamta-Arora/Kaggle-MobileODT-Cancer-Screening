#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:10:49 2017

@author: daniele
"""

import tensorflow as tf
import numpy as np

# =============================================================================
# ============================ FUNCTION FOR INPUT =============================
# =============================================================================


def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    return tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1],
                                       image_shape[2]], name="x")


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(tf.float32, [None, n_classes], name="y")


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    """
    return tf.placeholder(tf.float32, name="keep_prob")


def neural_net_learning_rate_input():
    """
    Return a Tensor for the learning rate
    """
    return tf.placeholder(tf.float64, name="learning_rate")


# =============================================================================
# ================== FUNCTION FOR CONVOLUTION AND MAX POOLING =================
# =============================================================================


def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides,
                   pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # Number of input colors
    num_inputcolors = x_tensor.shape.as_list()[3]

    # Convolutional filter
    W_conv = tf.Variable(tf.truncated_normal([conv_ksize[0], conv_ksize[1],
                                             num_inputcolors,
                                             conv_num_outputs], stddev=0.1))
    b_conv = tf.Variable(tf.constant(0.1, shape=[conv_num_outputs]))

    convolution = tf.nn.conv2d(x_tensor, W_conv, strides=[1, conv_strides[0],
                                                          conv_strides[1], 1],
                               padding='SAME')
    h_conv = tf.nn.relu(convolution + b_conv)

    h_pool = tf.nn.max_pool(h_conv, ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                            strides=[1, pool_strides[0], pool_strides[1], 1],
                            padding='SAME')

    return h_pool


def make_convolutional_layers(x, convolutional_layers, keep_prob):
    """
    Takes a list specifying parameters for convolutional layers and turns it
    into a list of correctly ordered TensorFlow convolutional layers.
    Each layer consists of a conv2d_maxpool and a dropout.
    Parameters:
        x: input tensor (e.g. the TensorFlow variable for the image array)
        convolutional_layers: list of data specifying each convolutional layer.
                              The for each element is
                              [int (number of output channels),
                               tuple of length 2 (size of conv filter),
                               tuple of length 2 (step size of conv filter),
                               tuple of length 2 (size of max pooling filter),
                               tuple of length 2 (step size max pooling filter)
                               ]
        keep_prob: TensorFlow variable specifying the dropout (tf.float32)
    """
    # Make the first layer.
    all_conv_layers = [tf.nn.dropout(
            conv2d_maxpool(x, convolutional_layers[0][0],
                           convolutional_layers[0][1],
                           convolutional_layers[0][2],
                           convolutional_layers[0][3],
                           convolutional_layers[0][4]),
                       keep_prob)]
    # Now for each additional element in convolutional_layers, add a new layer
    # which takes as input the previous layer.
    for ii, conv_lyr in enumerate(convolutional_layers):
        if ii > 0:
            all_conv_layers.append(tf.nn.dropout(
                                       conv2d_maxpool(all_conv_layers[ii-1],
                                                      conv_lyr[0], conv_lyr[1],
                                                      conv_lyr[2], conv_lyr[3],
                                                      conv_lyr[4]), keep_prob))
    return all_conv_layers


# =============================================================================
# ======================= FUNCTION FOR FLATTENING LAYER =======================
# =============================================================================


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image
                dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    flat_dimension = np.prod(x_tensor.shape.as_list()[1:])
    x_flat = tf.reshape(x_tensor, [-1, flat_dimension])

    return x_flat


# =============================================================================
# ==================== FUNCTION FOR FULLY CONNECTED LAYERS ====================
# =============================================================================


def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    input_dimensions = x_tensor.shape.as_list()[1]
    W = tf.Variable(tf.truncated_normal([input_dimensions, num_outputs],
                                        stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[num_outputs]))

    h_connected = tf.nn.relu(tf.matmul(x_tensor, W) + b)

    return h_connected


def make_fullyconnected_layers(flattened_tensor, connected_layers, keep_prob):
    """
    Takes a list specifying the output sizes of connected layers and outputs
    a list containing all the fully connected network layers.
    Parameters:
        flattened_tensor: the input tensor to the first fully connected layer
        connected_layers: a list containing output-layer sizes, e.g. [10, 20]
        keep_prob: TensorFlow variable specifying the dropout (tf.float32)
    """
    allconnectedlayers = [flattened_tensor]
    for ii, num_outputs in enumerate(connected_layers):
        allconnectedlayers.append(tf.nn.dropout(
                                           fully_conn(allconnectedlayers[ii],
                                                      num_outputs),
                                           keep_prob))
    return allconnectedlayers


# =============================================================================
# ======================== FUNCTION FOR OUTPUT LAYERS =========================
# =============================================================================


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    input_dimensions = x_tensor.shape.as_list()[1]
    W = tf.Variable(tf.truncated_normal([input_dimensions, num_outputs],
                                        stddev=0.1))
    b = tf.Variable(tf.constant(0.1, shape=[num_outputs]))

    h_output = tf.matmul(x_tensor, W) + b

    return h_output


# =============================================================================
# ================= FUNCTION FOR COST, OPTIMIZER AND ACCURACY =================
# =============================================================================


def make_cost_optimizer_accuracy(logits, labels, learning_rate):
    """
    Given logits, labels and a learning rate, returns the computed cost (using
    softmax and cross entropy) and accuracy, as well as the optimizer.
    Output: cost, optimizer, accuracy
    """
    # Loss and Optimizer
    cost = tf.reduce_mean(
               tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                       labels=labels))
    optimizer = tf.train.AdamOptimizer(
                    learning_rate=learning_rate, name='Adam'
                    ).minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32),
                              name='accuracy')

    return cost, optimizer, accuracy
