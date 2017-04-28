#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:26:31 2017

@author: daniele
"""

import os


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
    return savedmodel_path[:savedmodel_path.rfind("-")]
