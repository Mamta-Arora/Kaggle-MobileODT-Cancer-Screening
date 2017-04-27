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
