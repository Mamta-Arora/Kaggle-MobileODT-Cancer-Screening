#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 11:49:29 2017

@author: daniele
"""
import numpy as np
import matplotlib.pyplot as plt


def display_images(list_of_image_matrices, figsizes=(16, 4)):
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


def display_single_image(imagearray):
    """
    Displays a single image, given an image array, and prints out the image
    shape.
    """
    array_to_plot = imagearray
    print("Image shape: {}".format(imagearray.shape))
    plt.imshow(array_to_plot)
    plt.axis("off")
    plt.show()


def plot_accuracy_trainloss_valloss(accuracies, train_losses, val_losses):
    """
    Given the parameters accuracies, train_losses, val_losses, plots the
    accuracy and losses through the training epochs.
    """
    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.show()

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


def plot_best_scores(best_accuracy, best_train_loss, best_val_loss):
    """
    Given the best accuracy, training loss, validation loss for each training
    run (with increasing amount of data), plots the values.
    """
    plt.plot(best_accuracy)
    plt.xlabel("Number of batches in training")
    plt.ylabel("Best Accuracy")
    plt.show()

    plt.plot(best_train_loss)
    plt.plot(best_val_loss)
    plt.xlabel("Number of batches in training")
    plt.ylabel("Best loss")
    plt.show()
