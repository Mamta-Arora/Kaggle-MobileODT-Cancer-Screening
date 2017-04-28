#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:12:39 2017

@author: daniele
"""

import numpy as np


def generic_data_loader(batch_location, batch_number, image_numbers=[]):
    """
    Loads training data or labels from files. It is possible to specify an
    interval of images to load, or by defauly load the entire batch.
    Parameters:
        batch_location: string, specifying the path to the batches. Includes
                        the file name up to but excluding the batch number.
        batch_number: integer, specifying the batch number to be loaded
        image_numbers: list, specifying the indexes of the images in the batch
                       that are to be loaded.
    """
    if image_numbers == []:
        return np.load(batch_location + str(batch_number) + ".npy")
    else:
        return np.load(batch_location + str(batch_number) +
                       ".npy")[image_numbers]


def load_training_data(batch_number, image_numbers=[]):
    """
    Loads the training data from files. It is possible to specify an interval
    of images to load, or by defauly load th entire batch.
    """
    batch_location = "./TensorFlow_data/training_data/training_images_batch"
    return generic_data_loader(batch_location, batch_number,
                               image_numbers=image_numbers)


def load_training_labels(batch_number, image_numbers=[]):
    """
    Loads the training labels from files. It is possible to specify an interval
    of labels to load, or by defauly load th entire batch.
    """
    batch_location = "./TensorFlow_data/training_data/training_labels_batch"
    return generic_data_loader(batch_location, batch_number,
                               image_numbers=image_numbers)
