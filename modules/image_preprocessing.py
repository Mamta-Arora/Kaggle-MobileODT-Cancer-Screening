#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:05:08 2017

@author: daniele
"""
import scipy.ndimage
from joblib import Parallel, delayed
import multiprocessing
import numpy as np
from modules.data_loading import load_training_data, load_training_labels
from modules.neural_network import oversample
from modules.image_cropping_KAGGLECODE import crop_image


def one_hot_encode(list_of_types, encoder):
    """
    One hot encode a list of Types. Returns a one-hot encoded vector for each
    Type.
    """
    return encoder.transform(list_of_types)


def get_Type(filepath, image_safe=False, resize_shape=(150, 150, 3)):
    """
    Returns the type corresponding to an image found in filepath. If
    image_safe is set to True, we attempt to preproces the image (using the
    input resize_shape); this may fail, and so we only return the image Type if
    the preprocessing was a success.
    """
    # The type number is given by the name of the folder in which we find the
    # image
    indexname = filepath.rfind("/")
    letter = filepath[indexname-6:indexname]
    if image_safe is False:
        return letter
    else:
        imagearray = load_normalize_image(filepath, resize_shape=resize_shape)
        if imagearray is not None:
            # The preprocessing was successful
            return letter


def make_square(single_image_array):
    """
    Taken the array of an image and makes the image square by padding it with
    0-value pixels on either side of the center.
    Input and output are both numpy arrays describing an image
    """
    image_shape = single_image_array.shape
    if image_shape[0] > image_shape[1]:
        # Need to add columns to the image
        colstoadd_eachside = int((image_shape[0] - image_shape[1]) / 2.)
        square_image = np.pad(single_image_array, ((0, 0),
                                                   (colstoadd_eachside,
                                                    colstoadd_eachside),
                                                   (0, 0)), "constant")
    if image_shape[1] > image_shape[0]:
        # Need to add rows to the image
        rowstoadd_eachside = int((image_shape[1] - image_shape[0]) / 2.)
        square_image = np.pad(single_image_array, ((rowstoadd_eachside,
                                                    rowstoadd_eachside),
                                                   (0, 0), (0, 0)), "constant")
    return square_image


def load_normalize_image(path, resize_shape=(150, 150, 3), crop=False):
    """
    Takes the directory path of an image and returns a normalized
    3-dimensional array representing that image.
    """
    # First we load the image
    try:
        imagearray = scipy.ndimage.imread(path)
        # The images contain a lot that isn't the area. The following function
        # by a Kaggle kernel crops the image to the relevant area
        if crop:
            # Try and crop it. If there are problems, don't crop it.
            try:
                imagearray = crop_image(imagearray)
            except:
                pass
        # Now we make the image square
        imagearray = make_square(imagearray)
        # There is no need to reshape the image to be three-dimensional; they
        # already are. We do want to resize it however.
        imagearray = scipy.misc.imresize(imagearray, resize_shape)
        # Now we normalize it
        imagearray = imagearray / 255.
        return imagearray
    except:
        # If some images are broken in the database; these will raise errors.
        pass


def array_all_images(list_of_path_names, parallelize=False):
    """
    Takes a list of directory paths of images and returns a 4-dimensional array
    containing the pixel-data of those images. The shape is:
    (num_images, x_dim, y_dim, num_colors)
    """
    if parallelize:
        num_cores = multiprocessing.cpu_count()
        all_images = Parallel(n_jobs=num_cores)(
                delayed(load_normalize_image)(path)
                for path in list_of_path_names)
    else:
        all_images = [load_normalize_image(path)
                      for path in list_of_path_names]
    # Some of these might be None since the function load_normalize_image
    # does not load broken images. We now remove these Nones.
    all_images = [img for img in all_images if img is not None]
    # IN PYTHON 3 np.array(list(filter(None.__ne__, all_images)))
    return all_images


def array_all_labels(list_of_path_names, encoder, parallelize=False,
                     resize_shape=(150, 150, 3)):
    """
    Takes a list of directory paths of images and returns a 2-dimensional array
    containing the one-hot-encoded labels of those images
    """
    if parallelize:
        num_cores = multiprocessing.cpu_count()
        the_types = Parallel(n_jobs=num_cores)(
                delayed(get_Type)(path, image_safe=True,
                                  resize_shape=resize_shape)
                for path in list_of_path_names)
    else:
        the_types = [get_Type(path, image_safe=True, resize_shape=resize_shape)
                     for path in list_of_path_names]
    the_types = [typ for typ in the_types if typ is not None]
    # IN PYTHON 3: list(filter(None.__ne__, the_types))
    all_labels = one_hot_encode(the_types, encoder)
    return all_labels


def flip_leftright(input_arrays, input_labels):
    """
    Convience function for increasing the amount of data by flipping the
    images left-to-right. Returns the doubled-up imagearrays and their labels.
    """
    flipped_array = np.concatenate((input_arrays, input_arrays[:, :, ::-1]),
                                   axis=0)
    output_labels = np.concatenate((input_labels, input_labels), axis=0)
    return flipped_array, output_labels


def flip_updown(input_arrays, input_labels):
    """
    Convience function for increasing the amount of data by flipping the
    images upside-down. Returns the doubled-up imagearrays and their labels.
    """
    flipped_array = np.concatenate((input_arrays, input_arrays[:, ::-1]),
                                   axis=0)
    output_labels = np.concatenate((input_labels, input_labels), axis=0)
    return flipped_array, output_labels


def batch_load_manipulate(batch_number, leftright=True, updown=True):
    """
    Prepreocesses a batch of image arrays and their labels, by loading a batch
    and includes images that have been flipped left-to-right and upside-down,
    if specified by the function arguments. Also oversamples images to provide
    a balanced set to train on.
    Input:
        batch_number: int specifying the batch number
        leftright: booloean specifying whether to also include a flipped
                   version of the images or not
        updown: booloean specifying whether to also include a flipped
                version of the images or not
    Output:
        loaded_batch: the oversampled image array
        loaded_labels: the labels to loaded_batch
    """
    # Load the batch from disk
    loaded_batch = load_training_data(batch_number)
    loaded_labels = load_training_labels(batch_number)
    # If we also include images flipped left-to-right or
    # upside-down, we add these to batch_inputarray and
    # batch_labels (the labels don't change of course).
    if leftright:
        (loaded_batch, loaded_labels) = flip_leftright(loaded_batch,
                                                       loaded_labels)
    if updown:
        (loaded_batch, loaded_labels) = flip_updown(loaded_batch,
                                                    loaded_labels)
    # Finally, we need to resample the images so that the
    # different classes appear an equal number of times
    if oversample:
        (loaded_batch, loaded_labels) = oversample(loaded_batch, loaded_labels)
    return (loaded_batch, loaded_labels)


def mean_RGB(single_image_array):
    """
    Turns an image (in array form) into a single pixel, as the average of all
    the image's pixels. It then normalizes the pixels to sum to 1.
    Input: image 3-d array.
    Output: 1-d array describing the average (and mormalized) pixel.
    """
    mean_rgb_values = np.mean(np.mean(single_image_array, axis=0), axis=0)
    normalized_values = mean_rgb_values / np.sum(mean_rgb_values)
    return normalized_values


def images_to_mean_RGB(array_of_images):
    """
    Conveniece function that applies mean_RGB to all images in an array.
    """
    return np.array([mean_RGB(img_ar) for img_ar in array_of_images])
