#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:09:41 2017

@author: daniele
"""

# Import relevant packages
import numpy as np
import matplotlib.pyplot as plt
import os

from os.path import isfile

import scipy.ndimage
from sklearn.preprocessing import LabelBinarizer
from tqdm import tqdm

from modules.path_munging import all_image_paths, batch_list, create_folder
from modules.visualization import display_images
from modules.image_preprocessing import (get_Type, array_all_images,
                                         array_all_labels)


class DataPreprocessor(object):
    """
    Preprocessing object. Loads image files, transforms them to a preferred
    format, and saves them to disk.
    """

    def __init__(self, training_folders=[], testing_folder="", **kwargs):
        self.training_folders = training_folders
        self.testing_folder = testing_folder
        # Now go through the training and testing folders to make a list of
        # paths (strings) to the image files.
        # We first get all path-names for the training and testing images
        training_pathnames = sum([all_image_paths(folder)
                                  for folder in training_folders], [])
        testing_pathnames = all_image_paths(testing_folder)
        # In each folder all images depict the same cervical type
        all_Types = np.sort([get_Type(pathname[0])
                             for pathname in training_pathnames])
        # We may now make the function that one-hot-encodes Types into arrays.
        # This will be necessary later when we make arrays with one-hot-encoded
        # labels.
        self.alltypes = all_Types
        enc = LabelBinarizer()
        enc.fit(all_Types)
        self.encoder = enc
        # We now flatten the lists of path names
        training_pathnames = np.array(sum(training_pathnames, []))
        testing_pathnames = np.array(sum(testing_pathnames, []))
        # When training, we don't want the images to be ordered. Therefore, we
        # take a random permutation of their order. For reproducibility we fix
        # the random seed.
        np.random.seed(42)
        training_pathnames = np.random.permutation(training_pathnames)
        self.training_pathnames = training_pathnames
        self.testing_pathnames = testing_pathnames

    def test_loading(self, images_to_load=range(7, 13)):
        """
        Tests whether the images in the folder are loaded and display as
        expected.
        Parameters:
            images_to_load: list of indices specifying which training images
                            to load.
        """
        list_of_image_arrays = [scipy.ndimage.imread(impath)
                                for impath in
                                self.training_pathnames[images_to_load]]
        fig, axes = display_images(list_of_image_arrays)
        fig.suptitle('Examples of loaded images:', fontsize=30)
        plt.show(fig)

    def test_resizing(self, resize_shape=(150, 150, 3), index_image=17):
        """
        Checks whether a given image resizing is appropriate by displaying the
        resized image next to its original for comparison.
        Parameters:
            resize_shape: tuple, the new image array shape we size to.
            index_image: int, the index of the training images we check the
                         resizing on.
        """
        imagearray = scipy.ndimage.imread(self.training_pathnames[index_image])
        resized_imagearray = scipy.misc.imresize(imagearray, resize_shape)
        fig, axes = display_images([imagearray, resized_imagearray])
        fig.suptitle('Image resolution (before and after preprocessing):',
                     fontsize=20)
        plt.show(fig)

    def preprocess_save(self, data_folder="./TensorFlow_data",
                        training_subfolder="/training_data",
                        testing_subfolder="/testing_data", 
                        resize_shape=(150, 150, 3), batch_size=2**7,
                        parallelize=True):
        """
        If this has not already been done, preprocess_save preprocesses all the
        images, turning them into numpy arrays, and saves them to disk.
        """
        # If necessary, create the folders in which we'll place the
        # preprocessed numpy arrays.
        create_folder(data_folder)
        trainingarrays_folder = data_folder + training_subfolder
        testingarray_folder = data_folder + testing_subfolder
        create_folder(trainingarrays_folder)
        create_folder(testingarray_folder)

        # Make the input data for the TESTING set
        testingarrayimage_path = "/testing_images.npy"
        # testingarrayname_path is a list of image names,
        # e.g. ["./path/0.jpg", "./path/15.jpg", ...]
        testingimagename_path = "/testing_namelabels.npy"
        # categoryorder_path contains the order in which the Types are one-hot-
        # encoded, e.g. categoryorder_path=["Type_1", "Type_2", "Type_3"] means
        # that a label [0, 1, 0] is Type 2.
        categoryorder_path = "/type123_order.npy"
        if isfile(testingarray_folder + testingarrayimage_path) is False:
            print("Creating testing data arrays...")
            testing_images = array_all_images(self.testing_pathnames,
                                              parallelize=parallelize)
            if len(testing_images) != len(self.testing_pathnames):
                print("WARNING: SOME TESTING IMAGES WERE NOT STORED AS NUMPY "
                      "ARRAYS!")
            np.save(testingarray_folder + testingarrayimage_path,
                    testing_images)
            np.save(testingarray_folder + testingimagename_path,
                    self.testing_pathnames)
            np.save(testingarray_folder + categoryorder_path,
                    self.alltypes)
            print("Testing data arrays created")

        # Make the input data for the TRAINING set
        # We first turn training_pathnames into batches of pathnames.
        training_pathnames_batches = batch_list(self.training_pathnames,
                                                batch_size)
        num_saved_batches = sum(["training_images_batch" in filename
                                 for filename in list(
                                        os.walk(trainingarrays_folder))[0][2]])
        # If we have a different number of batches saved comapred to what we
        # want, the batches are wrong and need recomputing.
        if num_saved_batches != len(training_pathnames_batches):
            print("Creating training data arrays...")
            # We could delete the old files, but this is dangerous, since a
            # typo could remove all files on the computer. We simply overwrite
            # the files we have.
            for ii, batch in enumerate(tqdm(training_pathnames_batches)):
                training_images_batch = array_all_images(
                                                       batch,
                                                       parallelize=parallelize)
                np.save(trainingarrays_folder + "/training_images_batch" +
                        str(ii) + ".npy", training_images_batch)

                training_labels_batch = array_all_labels(
                                                       batch,
                                                       self.encoder,
                                                       parallelize=parallelize)
                np.save(trainingarrays_folder + "/training_labels_batch" +
                        str(ii) + ".npy", training_labels_batch)
            print("Training data arrays created")
