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

from modules.path_munging import all_image_paths, batch_list
from modules.visualization import drawSlices
from modules.image_preprocessing import (get_Type, array_all_images,
                                         array_all_labels)


# ===================== USER INPUT =====================
training_folders = ["./Data/train"]
# + ["./Data/Type_1", "./Data/Type_2", "./Data/Type_3"]
testing_folder = "./Data/test"
# ======================================================


# We first get all path-names for the training and testing images
training_pathnames = sum([all_image_paths(folder)
                          for folder in training_folders], [])
testing_pathnames = all_image_paths(testing_folder)

# In each folder all images depict the same cervical type
all_Types = np.sort([get_Type(pathname[0]) for pathname in training_pathnames])

# We may now make the function that one-hot-encodes Types into arrays.
# This will be necessary later when we make arrays with one-hot-encoded labels.
enc = LabelBinarizer()
enc.fit(all_Types)

# We now flatten the lists of path names
training_pathnames = np.array(sum(training_pathnames, []))
testing_pathnames = np.array(sum(testing_pathnames, []))

# When training, we don't want the images to be ordered. Therefore, we take a
# random permutation of their order. For reproducibility we fix the random seed
np.random.seed(42)
training_pathnames = np.random.permutation(training_pathnames)
print("We are training on {} images".format(len(training_pathnames)))
print("We are testing on {} images".format(len(testing_pathnames)))

fig, axes = drawSlices([scipy.ndimage.imread(impath)
                        for impath in training_pathnames[7:13]])
fig.suptitle('Examples of loaded images:', fontsize=30)
plt.show(fig)


# ===================== USER INPUT =====================
resize_shape = (150, 150, 3)
# ======================================================


imagearray = scipy.ndimage.imread(training_pathnames[17])
resized_imagearray = scipy.misc.imresize(imagearray, resize_shape)
fig, axes = drawSlices([imagearray, resized_imagearray])
fig.suptitle('Image resolution (before and after preprocessing):', fontsize=20)
plt.show(fig)


# ===================== USER INPUT =====================
tensorflow_folder = "./TensorFlow_data"
training_subfolder = "/training_data"
testing_subfolder = "/testing_data"
# ======================================================


# We store all the data in a training and testing folder
training_folder = tensorflow_folder + training_subfolder
testing_folder = tensorflow_folder + testing_subfolder
if not os.path.exists(training_folder):
    os.makedirs(training_folder)
if not os.path.exists(testing_folder):
    os.makedirs(testing_folder)

# Make the input data and labels for the testing set
testingarrayimage_path = "/testing_images.npy"
testingarrayname_path = "/testing_namelabels.npy"
categoryorder_path = "/type123_order.npy"
if isfile(testing_folder + testingarrayimage_path) is False:
    print("Creating testing data arrays...")
    testing_images = array_all_images(testing_pathnames, parallelize=True)
    if len(testing_images) != len(testing_pathnames):
        print("SOME TESTING IMAGES WERE NOT STORED AS NUMPY ARRAYS!")
    np.save(testing_folder + testingarrayimage_path, testing_images)
    np.save(testing_folder + testingarrayname_path, testing_pathnames)
    np.save(testing_folder + categoryorder_path, all_Types)
    print("Testing data arrays created")


# ===================== USER INPUT =====================
# Here we specify the size of each batch
batch_size = 2**7
# ======================================================


# Now we save the batch-data, unless it already exists
training_pathnames_batches = batch_list(training_pathnames, batch_size)
num_saved_batches = sum(["training_images_batch" in filename
                         for filename in list(os.walk(training_folder))[0][2]])

# If we have a different number of batches saved comapred to what we want,
# the batches are wrong and need recomputing.
if num_saved_batches != len(training_pathnames_batches):
    print("Creating training data arrays...")
    # We could delete the old files, but this is dangerous, since a typo could
    # remove all files on the computer. We simply overwrite the files we have.
    for ii, batch in enumerate(tqdm(training_pathnames_batches)):
        training_images_batch = array_all_images(batch, parallelize=True)
        np.save(training_folder + "/training_images_batch" + str(ii) + ".npy",
                training_images_batch)

        training_labels_batch = array_all_labels(batch, enc, parallelize=True)
        np.save(training_folder + "/training_labels_batch" + str(ii) + ".npy",
                training_labels_batch)
    print("Training data arrays created")
