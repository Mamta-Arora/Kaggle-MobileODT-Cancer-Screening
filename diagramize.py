# -*- coding: utf-8 -*-
"""
Created on Thu May  4 13:31:28 2017

@author: Sam
"""

# Based on http://scikit-learn.org/stable/auto_examples/cluster/plot_color_quantization.html#sphx-glr-auto-examples-cluster-plot-color-quantization-py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.utils import shuffle

def diagramify(imgarray, n_colors, cutoff = 0.33):
    masked = makemask(imgarray, cutoff = cutoff)
    diagram = ncolours(masked, n_colors)
    return diagram

def makemask(oneimage, cutoff = 0.33):
    n_colors = 2
    w, h, d = tuple(oneimage.shape)
    assert d == 3
    image_array = np.reshape(oneimage, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    newimage = mask(kmeans.cluster_centers_, labels, w, h, oneimage, cutoff = cutoff)
    return newimage

def ncolours(newimage, n_colors):
    w, h, d = tuple(newimage.shape)    
    assert d == 3
    image_array = np.reshape(newimage, (w * h, d))
    image_array_sample = shuffle(image_array, random_state=0)[:1000]
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
    labels = kmeans.predict(image_array)
    diagram = recreate_image(kmeans.cluster_centers_, labels, w, h)
    return diagram

def mask(codebook, labels, w, h, origimage, cutoff = 0.33):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            if max(image[i][j]) <= cutoff:
                # Turn that pixel black entirely
                image[i][j] = [0., 0., 0.]
            else:
                # Reset it to its original color
                image[i][j] = origimage[i][j]
            label_idx += 1
    return image

def recreate_image(codebook, labels, w, h):
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image