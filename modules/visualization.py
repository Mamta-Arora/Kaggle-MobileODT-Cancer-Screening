#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:49:29 2017

@author: daniele
"""
import numpy as np
import matplotlib.pyplot as plt


def drawSlices(list_of_image_matrices, figsizes=(16, 4)):
    """
    Makes subplots containing images drawn from the input arrays.
    Input: a list of matrices describing the pixels. The matrices are not
    required to be all have the same resolution or dimensions.
    Returns the tuple (fig, axes).
    """
    # We'll make a figure with subplots in it
    numberofrows = int(np.ceil(len(list_of_image_matrices) / 6.))
    fig, axes = plt.subplots(nrows=numberofrows, ncols=6,
                             figsize=(figsizes[0], figsizes[1]*numberofrows))
    # Now on each axis we can draw the slice
    for (currentax, currentimage) in zip(axes.ravel(), list_of_image_matrices):
        currentax.imshow(currentimage, cmap="gray")
        # The ticks are useless and ugly
        currentax.set_xticks([])
        currentax.set_yticks([])
    # Finally we remove those plots that have nothing in them
    for remainingax in axes.ravel()[len(list_of_image_matrices):]:
        remainingax.axis("off")
    return fig, axes
