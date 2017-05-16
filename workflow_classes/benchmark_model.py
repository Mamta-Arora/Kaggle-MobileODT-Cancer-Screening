#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:04:33 2017

@author: daniele
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from modules.data_loading import load_training_data
from modules.visualization import display_single_image
from modules.path_munging import count_batches
from modules.image_preprocessing import (batch_load_manipulate,
                                         images_to_percentage_red)
from modules.probablity_manipulation import compute_loss, agnosticize


class BenchmarkModel(object):
    """
    Random Forest training on average pixel color.
    """

    def __init__(self, model_name="", **kwargs):
        self.model_name = model_name

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

    def count_training_batches(self, folder):
        """
        Convenience function which counts the number of training-batch files
        in the specified folder.
        Input: string specifying the path to the folder.
        """
        return count_batches(folder)

    def train(self, training_batches=[], leftright=False, updown=False,
              validation_inputarray=[], validation_labels=[],
              validation_batchnum=0, agnosticic_average=0, training_folder=""):
        """
        Trains the random forest on images reduced to a single (average) pixel.
        It is possible to specify the validation set by either giving the
        function the number of a data batch, or by giving it the array of the
        validation data and the array of its labels.
        Parameters:
            training_batches: list of ints. Specifies the number-labels of the
                              batches to be used for training.
            leftright: boolean. Specifies whether we should also train on
                                images that have been flipped left-to-right.
            updown: boolean. Specifies whether we should also train on images
                             that have been flipped upside-down.
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
        Returns:
            accuracy: validation-set accuracy.
            train_loss: training-set loss.
            val_loss: validation-set loss.
        """
        # Load the training batches (oversampling to make sure they're
        # balanced) and concatenate them together
        all_train_data = []
        all_train_labels = []
        for batch_i in training_batches:
            (train_data,
             train_labels) = batch_load_manipulate(batch_i,
                                                   leftright=leftright,
                                                   updown=updown,
                                                   batch_loc=training_folder)
            all_train_data.append(train_data)
            all_train_labels.append(train_labels)
        all_train_data = np.concatenate(all_train_data)
        labels_train = np.concatenate(all_train_labels)

        # Load the validation set (again in a balanced way)
        if validation_inputarray != []:
            val_data = validation_inputarray
            labels_val = validation_labels
        else:
            (val_data,
             labels_val) = batch_load_manipulate(validation_batchnum,
                                                 leftright=False, updown=False,
                                                 batch_loc=training_folder)

        # Turn the input data for each image into a single number, which equals
        # the percentage of red pixels in the image
        input_train = images_to_percentage_red(all_train_data)
        input_val = images_to_percentage_red(val_data)

        # Fit random forest
        rand_forest = RandomForestClassifier(n_estimators=1000)
        rand_forest.fit(input_train, labels_train)
        self.trained_model = rand_forest

        # Compute accuracy, training loss and validation loss
        (accuracy,
         train_loss,
         val_loss) = self.get_stats(input_train, labels_train, input_val,
                                    labels_val,
                                    agnosticic_average=agnosticic_average)

        return (accuracy, train_loss, val_loss)

    def compute_probas(self, clf, inputdata):
        """
        Takes a classifier and an input and predicts one-hot-encoded
        proabilities.
        """
        probabilities = np.transpose(np.array(
                                        clf.predict_proba(inputdata))[:, :, 1])
        return probabilities

    def compute_score(self, clf, inputdata, correctlabels):
        """
        Computes the probabilities assigned to the classes of the input data,
        picks the likeliest one, and checks how many times on average it agrees
        with the  correctlabels.
        """
        predicted_probas = self.compute_probas(clf, inputdata)
        argmax_probas = np.array([np.argmax(pp) for pp in predicted_probas])
        argmax_truelabels = np.array([np.argmax(pp) for pp in correctlabels])

        score = np.mean(argmax_probas == argmax_truelabels)
        return score

    def get_stats(self, training_inputarray, training_labels,
                  validation_inputarray, validation_labels,
                  agnosticic_average=0):
        """
        Obtain information about loss and validation accuracy
        : training_inputarray: Batch of Numpy image data
        : training_labels: Batch of Numpy label data
        : validation_inputarray: Batch of Numpy image data
        : validation_labels: Batch of Numpy label data
        """
        # Predict probabilites
        train_probas = self.compute_probas(self.trained_model,
                                           training_inputarray)
        val_probas = self.compute_probas(self.trained_model,
                                         validation_inputarray)
        if agnosticic_average > 0:
            train_probas = agnosticize(train_probas, agnosticic_average)
            val_probas = agnosticize(val_probas, agnosticic_average)

        # Compute accuracy, training loss and validation loss
        accuracy = self.compute_score(self.trained_model,
                                      validation_inputarray,
                                      validation_labels)
        train_loss = compute_loss(train_probas, training_labels)
        val_loss = compute_loss(val_probas, validation_labels)
        return accuracy, train_loss, val_loss

    def test(self, load_test_set="", test_set=[], agnosticic_average=0):
        """
        Makes predictions on a given set of data. Is able to average it out
        with an agnostic probability to obtain less confident probability
        estimates.
        Parameters:
            load_test_set: string. Only needs to be specified if test_set is
                           not. Full path to the .npy data containing the test
                           set arrays to be fed into the network.
            test_set: array. Only needs to be specified if load_test_set is
                      not. This is the test-set aray to be fed into the neural
                      network to obtain predicted probabilities for each label.
            agnosticic_average: int. Specifies how many times we average the
                                predicted probabilities with an agnostic
                                probability. Default is 0.
        Returns: probabilities.
        """
        if test_set == []:
            testing_inputarray = np.load(load_test_set)
        else:
            testing_inputarray = test_set

        input_test = images_to_percentage_red(testing_inputarray)
        probabilities = self.compute_probas(self.trained_model, input_test)
        if agnosticic_average > 0:
            probabilities = agnosticize(probabilities, agnosticic_average)

        return probabilities
