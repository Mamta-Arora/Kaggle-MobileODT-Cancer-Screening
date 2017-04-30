#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:26:31 2017

@author: daniele
"""

import os
import pandas as pd
import numpy as np


def all_image_paths(folderpath):
    """
    Returns a list of filenames containing 'jpg'. The returned list has
    sublists with filenames, where each sublist is a different folder.
    """
    image_pathnames = [[folderandfiles[0]+"/"+imname
                        for imname in folderandfiles[2] if "jpg" in imname]
                       for folderandfiles in os.walk(folderpath)
                       if folderandfiles[2] != []]
    image_pathnames = [folder for folder in image_pathnames if folder != []]
    return image_pathnames


def batch_list(inputlist, batch_size):
    """
    Returns the inputlist split into batches of maximal length batch_size.
    Each element in the returned list (i.e. each batch) is itself a list.
    """
    list_of_batches = [inputlist[ii: ii+batch_size]
                       for ii in range(0, len(inputlist), batch_size)]
    return list_of_batches


def create_folder(path_to_folder):
    """
    If the path_to_folder does not point to an existing folder, this function
    creates such a folder.
    """
    if not os.path.exists(path_to_folder):
        os.makedirs(path_to_folder)


def count_batches(folderpath):
    """
    Returns the number of training data batches in the input folder.
    Input: string specifying path
    Returns: int specifying the number of batches
    """
    return sum(["training_images_batch" in filename
                for filename in list(os.walk(folderpath))[0][2]])


def get_next_epoch(savedmodel_path):
    """
    Read the path to the saved model and returns the next eopch to train on.
    Input: string specifying path
    Returns: int specifying the next epoch
    """
    if savedmodel_path == "":
        next_epoch = 1
    else:
        next_epoch = int(savedmodel_path[savedmodel_path.rfind("-")+1:]) + 1
    return next_epoch


def get_modelpath_and_name(savedmodel_path):
    """
    Help function which returns the full path to a model, excluding the epoch
    number at the end, e.g. "./mybestmodel-40" returns "./mybestmodel".
    """
    return savedmodel_path[:savedmodel_path.rfind("-")]


def image_index(pathname):
    """
    Helper function for submission. Takes the path to an image, e.g.
    "./image_folder/15.jpg", and returns the image name, in this example
    "15.jpg".
    """
    return pathname[pathname.rfind("/")+1:]


def get_date_string():
    """
    Returns the current date and time in the format "yyyy-mm-dd_hh-mm".
    For example, 30 April 2017 at 16:40 is returned as '2017-04-30_16-40'.
    """
    currentime = pd.datetime.now().isoformat()
    # Format the time
    dateandminute = currentime[:currentime.rfind(":")
                               ].replace("T", "_").replace(":", "-")
    return dateandminute


def submission(probabilities, testing_folder, submission_folder):
    """
    Creates a csv submission file from predicted probabilities, compatible with
    Kaggle's submission guidelines. The file has the name submissions followed
    by the date and time of the file creating, i.e.
    "submissions_yyyy-mm-dd_hh-mm.csv".
    Parameters:
        probabilities: array of predicted probabilities for each image
        testing_folder: string specifying the folder containing the
                        testing-input data. From this folder we fetch the image
                        name labels (e.g. "15.jpg") and the name of the
                        classification cateogries.
        submission_folder: string specifying the folder into which we save the
                           submission csv.
    Returns: string specifying the full path of the csv file we have saved.
    """
    create_folder(submission_folder)

    # Get the list of image names ["15.jpg", "42.jpg", "1.jpg", ...]
    image_names = np.load(testing_folder + "/testing_namelabels.npy")
    image_names = [image_index(path) for path in image_names]
    # Get the order of the catogeries ["Type_1", "Type_2", "Type_3"]
    categories = np.load(testing_folder + "/type123_order.npy")

    # Make a dataframe containing the information
    submission_df = pd.DataFrame(probabilities, columns=categories)
    submission_df["image_name"] = image_names
    submission_df.set_index("image_name", inplace=True)

    filename = submission_folder + "/submissions_" + get_date_string() + ".csv"
    # Now save to csv
    submission_df.to_csv(filename)
    return filename
