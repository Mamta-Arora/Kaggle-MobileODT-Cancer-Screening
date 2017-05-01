#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 21:42:57 2017

@author: daniele
"""

import tensorflow as tf
import numpy as np
from modules.data_loading import load_training_data, load_training_labels
from modules.visualization import display_single_image
from modules.neural_network import (neural_net_image_input,
                                    neural_net_label_input,
                                    neural_net_keep_prob_input,
                                    neural_net_learning_rate_input,
                                    make_convolutional_layers,
                                    flatten,
                                    make_fullyconnected_layers,
                                    output,
                                    make_cost_optimizer_accuracy,
                                    oversample)
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
        self.logits = logits

        y = neural_net_label_input(output_channels)
        self.y = y
        learn_rate_variable = neural_net_learning_rate_input()
        self.learn_rate_variable = learn_rate_variable
        (cost,
         optimizer,
         accuracy) = make_cost_optimizer_accuracy(logits, y,
                                                  learn_rate_variable)
        self.cost = cost
        self.optimizer = optimizer
        self.accuracy = accuracy

    def test_loading(self, batch_and_index=(0, 19), batch_loc=""):
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
                                      batch_and_index[0],
                                      batch_loc=batch_loc)[batch_and_index[1]])

    def conv_net(self, input_shape, output_channels, convolutional_layers,
                 connected_layers):
        """
        Creates a convolutional neural network based on the given
        specifications. Uses a tensorflow variable for training and saves it
        as self.keep_prob_variable. Also saves the input array variable as
        self.x.
        Parameters:
            input_shape: tuple of length 3 specifying the size of each
                         dimension of the input.
            output_channels: int specifying the number of output channels, i.e.
                             the number of classification categories.
            convolutional_layers: list where each element specifies the
                                  parameters of a convolution + max pooling
                                  layer. See function make_convolutional_layers
                                  for format.
            connected_layers: a list containing full-connected-layer sizes,
                              e.g. [10, 20].
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
                                                 self.keep_prob_variable: 1.0,
                                                 self.learn_rate_variable:
                                                 self.learning_rate})
        validation_cost_value = session.run(self.cost,
                                            feed_dict={
                                                 self.x: validation_inputarray,
                                                 self.y: validation_labels,
                                                 self.keep_prob_variable: 1.0,
                                                 self.learn_rate_variable:
                                                 self.learning_rate})
        accuracy_value = session.run(self.accuracy,
                                     feed_dict={self.x: validation_inputarray,
                                                self.y: validation_labels,
                                                self.keep_prob_variable: 1.0,
                                                self.learn_rate_variable:
                                                self.learning_rate})
        if printout:
            print("\nTraining Loss: {}".format(training_cost_value))
            print("Validation Loss: {}".format(validation_cost_value))
            print("Accuracy (validation): {}".format(accuracy_value))
        return training_cost_value, validation_cost_value, accuracy_value

    def count_training_batches(self, folder):
        """
        Convenience function which counts the number of training-batch files
        in the specified folder.
        Input: string specifying the path to the folder.
        """
        return count_batches(folder)

    def train(self, epochs=10, load_saved_model="", training_batches=[],
              leftright=False, updown=False, size_of_minibatch=2**6,
              validation_inputarray=[], validation_labels=[],
              validation_batchnum=0, printout=True, save_model=True,
              model_destination_folder=""):
        """
        Trains the neural network. It is possible to specify the validation set
        by either giving the function the number of a data batch, or by giving
        it the array of the validation data and the array of its labels.
        Parameters:
            epochs: int. Number of training epochs.
            load_saved_model: string. Full path to the saved model, including
                              epoch number, e.g. "./bestmodel-40". Should be
                              set to "" if no model is to be loaded.
            training_batches: list of ints. Specifies the number-labels of the
                              batches to be used for training.
            leftright: boolean. Specifies whether we should also train on
                                images that have been flipped left-to-right.
            updown: boolean. Specifies whether we should also train on images
                             that have been flipped upside-down.
            size_of_minibatch: int. Number of image arrays to be used in each
                               model-optimization round.
            validation_inputarray: 4-d array. Only needs to be specified if
                                   validation_batchnum is not. The array is the
                                   input-data of our validation set.
            validation_labels: 2-d array. Only needs to be specified if
                               validation_batchnum is not. The array is the
                               one-hot-encoded labels of our validation set.
            validation_batchnum: int. Only needs to be specified if both
                                 validation_inputarray and validation_labels
                                 are not. Specifies the number-label of the
                                 data batch we want to use as our validation
                                 set.
            printout: boolean. Sets whether to include stats-printouts during
                      training. Will then print out validation-set accuracy,
                      training-set loss and validation-set loss.
            save_model: boolean. Sets whether to save intermediate trainined
                        models as well as the final trained model.
            model_destination_folder: string. If save_model is set to True,
                                      this parameter is required to determine
                                      the destination folder in which we save
                                      the trained models.
        Returns:
            accuracy_list: list of validation-set accuracies at the end of each
                           training epoch.
            training_losses: list of training-set loss at the end of each
                             training epoch.
            validation_losses: list of validation-set loss at the end of each
                               training epoch.
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
            saver = tf.train.Saver(max_to_keep=None)

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
                    # If we also include images flipped left-to-right or
                    # upside-down, we add these to batch_inputarray and
                    # batch_labels (the labels don't change of course).
                    if leftright:
                        batch_inputarray = np.concatenate(
                                                (batch_inputarray,
                                                 batch_inputarray[:, :, ::-1]),
                                                axis=0)
                        batch_labels = np.concatenate((batch_labels,
                                                       batch_labels), axis=0)
                    if updown:
                        batch_inputarray = np.concatenate(
                                                   (batch_inputarray,
                                                    batch_inputarray[:, ::-1]),
                                                   axis=0)
                        batch_labels = np.concatenate((batch_labels,
                                                       batch_labels), axis=0)
                    # Finally, we need to resample the images so that the
                    # different classes appear an equal number of times
                    (batch_inputarray,
                     batch_labels) = oversample(batch_inputarray, batch_labels)
                    for minibatch_inputarrays, minibatch_labels in zip(
                                               batch_inputarray, batch_labels):
                        # Train the network on a minibatch of data
                        sess.run(self.optimizer,
                                 feed_dict={
                                      self.x: minibatch_inputarrays,
                                      self.y: minibatch_labels,
                                      self.keep_prob_variable: self.keep_prob,
                                      self.learn_rate_variable:
                                      self.learning_rate})
                    # Evaluate how well the network is currently doing
                    (training_cost_value,
                     validation_cost_value,
                     accuracy_value) = self.get_stats(sess,
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

    def test(self, load_saved_model="", load_test_set="", test_set=[]):
        """
        Uses a saved pre-trained model to make predictions on a given set of
        data.
        Parameters:
            load_saved_model: string. Full path to the saved model, including
                              epoch number, e.g. "./bestmodel-40". Should be
                              set to "" if no model is to be loaded.
            load_test_set: string. Only needs to be specified if test_set is
                           not. Full path to the .npy data containing the test
                           set arrays to be fed into the network.
            test_set: array. Only needs to be specified if load_test_set is
                      not. This is the test-set aray to be fed into the neural
                      network to obtain predicted probabilities for each label.
        Returns: probabilities, obtained by applying a softmax function to the
                 logits.
        """
        if test_set == []:
            testing_inputarray = np.load(load_test_set)
        else:
            testing_inputarray = test_set

        with tf.Session() as sess:
            saver = tf.train.Saver(max_to_keep=None)
            saver.restore(sess, load_saved_model)

            # We need to supply a y. Since it plays no role when predicting
            # probabilities, we supply an  empty matrix with the right
            # dimensions.
            empty_y = np.zeros((testing_inputarray.shape[0],
                                testing_inputarray.shape[-1]))
            # Ale the learning rate is unimportant here.
            probabilities = sess.run(tf.nn.softmax(self.logits),
                                     feed_dict={
                                         self.x: testing_inputarray,
                                         self.y: empty_y,
                                         self.keep_prob_variable: 1.0,
                                         self.learn_rate_variable:
                                         self.learning_rate})
        return probabilities
