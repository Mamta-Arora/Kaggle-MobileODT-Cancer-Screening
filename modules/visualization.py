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

    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
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

    plt.plot(best_train_loss, label="Training loss")
    plt.plot(best_val_loss, label="Validation loss")
    plt.xlabel("Number of batches in training")
    plt.ylabel("Best loss")
    plt.legend()
    plt.show()


def type_labels(prob, categories):
    """
    Helper function to display_image_probabilities.
    Input:
        prob: a 1-d list describing the one-hot encoded predicted probabilities
              of an image.
        categories: a 1-d list containing the classification labels that each
                    element in prob refers to.
    Returns a string formatted to relate each category to its corresponding
    probability, e.g. "Type 1: 0.1\nType 2:0.6\nType 3:0.3".
    """
    type_prob_list = [(categ.replace("_", " ") + ": " +
                       str(np.round(p_i, decimals=3)) + "\n")
                      for p_i, categ in zip(prob, categories)]
    return "".join(type_prob_list)[:-1]


def display_image_probabilities(list_of_image_matrices, probabilities,
                                testing_folder, figsizes=(16, 4)):
    """
    Makes subplots containing images drawn from the input arrays, along with
    the predicted probabilities that the images should be in each class.
    Input:
        list_of_image_matrices: a list of matrices describing the pixels.
        probabilities: a list of probabilities, where each element is a list
                       containing the probability for each class (one-hot
                       encoding).
        testing_folder: the path to the folder containing the testing images.
    Returns the tuple (fig, axes).
    """
    # Get the classification labels that one-hot-encoded probability refers to
    categories = np.load(testing_folder + "/type123_order.npy")

    # We'll make a figure with subplots in it
    numberofrows = int(np.ceil(len(list_of_image_matrices) / 6.))
    fig, axes = plt.subplots(nrows=numberofrows, ncols=6,
                             figsize=(figsizes[0], figsizes[1]*numberofrows))
    # Now on each axis we can draw the image with its probabilities
    for (currentax, currentimage, prob) in zip(axes.ravel(),
                                               list_of_image_matrices,
                                               probabilities):
        currentax.imshow(currentimage, cmap="gray")
        # The ticks are useless and ugly
        currentax.set_xticks([])
        currentax.set_yticks([])
        currentax.set_xlabel(type_labels(prob, categories), size=14)
    # Finally we remove those plots that have nothing in them
    for remainingax in axes.ravel()[len(list_of_image_matrices):]:
        remainingax.axis("off")
    return fig, axes


def darkBrewerColors(listlength=6):
    """
    Returns a list of dark Brewer-scale colors.
    """
    return np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a",
                     "#b15928"][:min(6, listlength)])


def lightBrewerColors(listlength=12):
    """
    Returns a list of light Brewer-scale colors.
    """
    return np.array(["#a6cee3", "#b2df8a", "#fb9a99", "#fdbf6f", "#cab2d6",
                     "#ffff99"][:min(6, listlength)])
