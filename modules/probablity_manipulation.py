#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 12:12:53 2017

@author: daniele
"""

import numpy as np


def minmaxlog(value):
    """
    Rounds the value as max(min(value,1−10^{-15},10^{-15})), wrapped by the
    natural logarithm. This is useful for calculating the cross-entropy on
    (regulated) probabilites.
    """
    return np.log(max(min(value, 1-10**(-15)), 10**(-15)))


def repeat_array(array_to_repeat, num_repeats):
    """
    Helper function to oversample_array. Concatenates array_to_repeat to
    itself, num_repeats times.
    """
    return np.concatenate(tuple([array_to_repeat
                                 for kk in range(num_repeats)]))


def agnosticize_single_prob(probs, times):
    """
    Average probability with completely agnostic probabilities, a
    user-specified number of times.
    Input:
        probs: 1-d array of probabilities of a single datapoint to average with
               agnostic probabilities.
        times: number of times to average
    Returns:
        mean_probs: the averaged probabilities
    """
    agnostic_prob = np.full(len(probs), 1. / len(probs))
    probs_to_average = np.concatenate((repeat_array([agnostic_prob], times),
                                       [probs]))
    mean_probs = average_probabilities(probs_to_average)
    return mean_probs


def agnosticize(probs, times):
    """
    Averages an array of probabilities with the agnostic probabilitiy, a
    user-specified number of times.
    """
    return np.array([agnosticize_single_prob(pp, times) for pp in probs])


def average_probabilities(list_of_probabilities):
    """
    Convenienvce function for taking a list of of probabilities of the form
    [probabilities1, probabilities2,...] and returns their average probability.
    """
    return np.mean(list_of_probabilities, axis=0)


def multiply_probabilities(list_of_probabilities):
    """
    Multiplies label-probabilities together elementwise, and then normalizes
    them. E.g. two input-probabilities, each with a single element,
    [[[0.1, 0.2, 0.7]], [[0.8, 0.1, 0.1]]] gives [[0.47059, 0.11765, 0.41176]].
    """
    multiplied_probs = reduce(lambda x, y: x*y, list_of_probabilities)
    sum_probs = np.sum(multiplied_probs, axis=1)
    normalized_probs = np.array([prob / psum
                                 for prob, psum in zip(multiplied_probs,
                                                       sum_probs)])
    return normalized_probs


def compute_loss(probabilities, labels):
    """
    Comutes the mean cross-entropy loss given probabilities and the correct
    labels.
    """
    vlog = np.vectorize(minmaxlog)
    log_probas = vlog(probabilities)
    loss = -np.mean([np.dot(pp, lab) for pp, lab in zip(log_probas, labels)])
    return loss
