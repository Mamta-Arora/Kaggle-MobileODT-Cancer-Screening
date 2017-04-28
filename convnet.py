#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 21:42:57 2017

@author: daniele
"""

import tensorflow as tf
from modules.data_loading import load_training_data, load_training_labels
from modules.visualization import display_single_image
from modules.neural_network import (neural_net_image_input,
                                    neural_net_label_input,
                                    neural_net_keep_prob_input,
                                    make_convolutional_layers,
                                    flatten,
                                    make_fullyconnected_layers,
                                    output,
                                    make_cost_optimizer_accuracy)
from modules.path_munging import (batch_list, count_batches, get_next_epoch,
                                  get_modelpath_and_name, create_folder)


class ConvNet(object):
    """
    Convolutional neural network.
    """

    def __init__(self, input_shape=(1, 1, 1), output_channels=10,
                 convolutional_layers=[], connected_layers=[], keep_prob=0.5,
                 learning_rate=0.001, model_name="", **kwargs):
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.model_name = model_name

        logits = self.conv_net(input_shape, output_channels,
                               convolutional_layers, connected_layers)

        y = neural_net_label_input(output_channels)
        self.y = y
        (cost,
         optimizer,
         accuracy) = make_cost_optimizer_accuracy(logits, y, learning_rate)
        self.cost = cost
        self.optimizer = optimizer
        self.accuracy = accuracy

    def test_loading(self, batch_and_index=(0, 19)):
        """
        Tests whether the images in the folder are loaded and display as
        expected.
        Parameters:
            image_to_load: tuple of length 2, each entry is an integer
                           Specifies the batch number and the index within that
                           batch of that image that is to be loaded and
                           visualized.
        """
        display_single_image(load_training_data(
                                       batch_and_index[0])[batch_and_index[1]])

    def conv_net(self, input_shape, output_channels, convolutional_layers,
                 connected_layers):
        """
        DESCRIPTION
        """
        # Remove previous weights, bias, inputs, etc..
        tf.reset_default_graph()

        # Inputs
        x = neural_net_image_input(input_shape)
        self.x = x
        # y = neural_net_label_input(output_channels)
        keep_prob_variable = neural_net_keep_prob_input()
        self.keep_prob_variable = keep_prob_variable

        conv_layers = make_convolutional_layers(x, convolutional_layers,
                                                keep_prob_variable)
        flattened_tensor = flatten(conv_layers[-1])
        fullconn_layers = make_fullyconnected_layers(flattened_tensor,
                                                     connected_layers,
                                                     keep_prob_variable)
        logits = output(fullconn_layers[-1], output_channels)
        # Name logits Tensor, so that is can be loaded from disk after training
        logits = tf.identity(logits, name='logits')

        return logits

    def set_learning_rate(self, learning_rate):
        """
        DESCRIPTION
        """
        self.learning_rate = learning_rate
        # Should perhaps name the optimizer "Adam", but probably not necessary
        self.optimizer = tf.train.AdamOptimizer(
                         learning_rate=learning_rate
                         ).minimize(self.cost)

    def get_stats(self, session, training_inputarray, training_labels,
                  validation_inputarray, validation_labels, printout=True):
        """
        Obtain information about loss and validation accuracy
        : session: Current TensorFlow session
        : training_inputarray: Batch of Numpy image data
        : training_labels: Batch of Numpy label data
        : cost: TensorFlow cost function
        : accuracy: TensorFlow accuracy function
        """
        training_cost_value = session.run(self.cost,
                                          feed_dict={
                                                 self.x: training_inputarray,
                                                 self.y: training_labels,
                                                 self.keep_prob_variable: 1.0})
        validation_cost_value = session.run(self.cost,
                                            feed_dict={
                                                 self.x: validation_inputarray,
                                                 self.y: validation_labels,
                                                 self.keep_prob_variable: 1.0})
        accuracy_value = session.run(self.accuracy,
                                     feed_dict={self.x: validation_inputarray,
                                                self.y: validation_labels,
                                                self.keep_prob_variable: 1.0})
        if printout:
            print("\nTraining Loss: {}".format(training_cost_value))
            print("Validation Loss: {}".format(validation_cost_value))
            print("Accuracy (validation): {}".format(accuracy_value))
        return training_cost_value, validation_cost_value, accuracy_value

    def count_training_batches(self, folder):
        return count_batches(folder)

    def train(self, epochs=10, load_saved_model="", training_batches=[],
              size_of_minibatch=2**6, validation_inputarray=[],
              validation_labels=[], validation_batchnum=0, printout=True,
              save_model=True, model_destination_folder=""):
        """
        DESCRIPTION
        load_saved_model: path to the model
        CAN SPECIFY THE VALIDATION SET EITHER BY GIVING IT EXPLIOCITLY OR BY
        GIVING IT A BATCH NUMBER
        """
        # Make the validation set#
        if validation_inputarray != []:
            val_array = validation_inputarray
            val_labels = validation_labels
        else:
            val_array = load_training_data(validation_batchnum)
            val_labels = load_training_labels(validation_batchnum)

        if printout:
            print('Training...')

        # We will store the accuracy and the training & validation loss during
        # training.
        accuracy_list = []
        training_losses = []
        validation_losses = []

        next_epoch = get_next_epoch(load_saved_model)

        with tf.Session() as sess:
            # It is very important the saver is defined INSIDE the block
            # "with tf.Session() as sess"
            # otherwise it will be very difficult to load the graph (unless we
            # name all the variables etc).
            saver = tf.train.Saver()

            # Initializing the variables or load them from a saved model
            if load_saved_model == "":
                sess.run(tf.global_variables_initializer())
                if save_model:
                    create_folder(model_destination_folder)
                    savedmodel_path = (model_destination_folder + "/" +
                                       self.model_name)
            else:
                saver.restore(sess, load_saved_model)
                if model_destination_folder == "":
                    # We save model in the folder from which we load
                    savedmodel_path = get_modelpath_and_name(load_saved_model)
                else:
                    create_folder(model_destination_folder)
                    savedmodel_path = (model_destination_folder + "/" +
                                       self.model_name)

            # Training cycle
            # In each eopch, go over all the specified batches. In each batch,
            # need to go over all minibatches.
            for epoch in range(next_epoch, next_epoch + epochs):
                for batch_i in training_batches:
                    # Load the batch from disk and split it up into minibatches
                    # according to size_of_minibatch.
                    batch_inputarray = batch_list(load_training_data(batch_i),
                                                  size_of_minibatch)
                    batch_labels = batch_list(load_training_labels(batch_i),
                                              size_of_minibatch)
                    for minibatch_inputarrays, minibatch_labels in zip(
                                               batch_inputarray, batch_labels):
                        # Train the network on a minibatch of data
                        sess.run(self.optimizer,
                                 feed_dict={
                                      self.x: minibatch_inputarrays,
                                      self.y: minibatch_labels,
                                      self.keep_prob_variable: self.keep_prob})
                        # Evaluate how well the network is currently doing
                        (training_cost_value,
                         validation_cost_value,
                         accuracy_value) = self.get_stats(
                                                         sess,
                                                         minibatch_inputarrays,
                                                         minibatch_labels,
                                                         val_array,
                                                         val_labels,
                                                         printout=printout)
                    if printout:
                        print('Epoch {:>2}, Batch {} '
                              'complete'.format(epoch, batch_i))
                training_losses.append(training_cost_value)
                validation_losses.append(validation_cost_value)
                accuracy_list.append(accuracy_value)

                if (epoch % 10 == 0) and save_model:
                    # Save the intermediate model
                    save_path = saver.save(sess, savedmodel_path,
                                           global_step=epoch)

            if save_model:
                # Save the final model
                save_path = saver.save(sess, savedmodel_path,
                                       global_step=epoch)
        return accuracy_list, training_losses, validation_losses
