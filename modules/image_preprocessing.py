#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 12:05:08 2017

@author: daniele
"""
import scipy.ndimage
from joblib import Parallel, delayed
import multiprocessing


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


def load_normalize_image(path, resize_shape=(150, 150, 3)):
    """
    Takes the directory path of an image and returns a normalized
    3-dimensional array representing that image.
    """
    # First we load the image
    try:
        imagearray = scipy.ndimage.imread(path)
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
