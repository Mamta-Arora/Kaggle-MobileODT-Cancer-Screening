#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:35:40 2017

@author: daniele
"""

from basic_preprocessing import DataPreprocessor
from modules.image_preprocessing import (path_to_meanRGB_and_red_pixels,
                                         get_Type)
from modules.path_munging import create_folder
from os.path import isfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from modules.visualization import darkBrewerColors
from mpl_toolkits.mplot3d import Axes3D  # This is used but not explicitly


class DataExplorator(DataPreprocessor):
    """
    Data exploration object. Loads image files, analyzes their RGBs, and plots
    the results.
    """

    def __init__(self, **kwargs):
        self.data_path = "./ExploratoryAnalysis/mean_RGBs_and_red_ratios.npy"
        create_folder("./ExploratoryAnalysis")
        super(DataExplorator, self).__init__(**kwargs)

    def create_pixel_summary_data(self):
        """
        For each training image, filters out the nearly-black pixels and
        computes the mean RGB value of the remaining pixels. It also
        additionally counts the percentage of "red" pixels in the image. This
        information is then saved to disk in a folder "./ExploratoryAnalysis".
        """
        # Turn the unordered list of pathnames to the images into separate
        # lists, grouped by which type they are
        pathnames_bytype = [[path for path in self.training_pathnames
                             if get_Type(path) == typ]
                            for typ in self.alltypes]
        # Turn each image into an average RGB and percentage of red pixels.
        # Then save to disk. Only do the hard work if it hasn't already been
        # done.
        if isfile(self.data_path) is False:
            # For each image, the format is
            # ([mean_R, mean_G, mean_B], percentage_red_pixels)
            # We have an array of such tuples for each classfication type
            mean_RGBs_and_red_ratios = [[path_to_meanRGB_and_red_pixels(path)
                                         for path in typXpaths]
                                        for typXpaths in pathnames_bytype]
            np.save(self.data_path, mean_RGBs_and_red_ratios)
        else:
            print("The data exploration file already exists in the folder "
                  "ExploratoryAnalysis; did not recompute it.")

    def mean_RGB_fromdata(self, typXRGBsRed):
        """
        Convenience function, which takes a list of (mean_RGB, percentage_red)
        for each image of a given classification type, and returns the mean RGB
        values for all images of that classification type.
        """
        allRGBs = np.transpose([RGBred[0] for RGBred in typXRGBsRed])
        mean_R_G_B = [np.mean(color) for color in allRGBs]
        return mean_R_G_B

    def plot_mean_RGBs(self):
        """
        Plots the mean RGB values of the different types of cervix-images.
        """
        mean_RGBs_and_red_ratios = np.load(self.data_path)
        # Now extract the information of the mean_RGBs for each image, and for
        # each cervix type compute the mean of that.
        mean_RGB_bytype = [self.mean_RGB_fromdata(typXRGBsRed)
                           for typXRGBsRed in mean_RGBs_and_red_ratios]

        # The easiest way to plot this is by plugging it into a dataframe and
        # use Pandas plotting
        df_to_plot = pd.DataFrame(np.transpose(mean_RGB_bytype),
                                  index=["R", "G", "B"],
                                  columns=self.alltypes)
        fig = df_to_plot.plot.bar()
        fig.set_title("Average pixel intensity for the different cervix Types")
        fig.set_xlabel("Pixel color")
        fig.set_ylabel("Color intensity")
        plt.show(fig)

    def plot_RGB_scatter(self):
        """
        Plots a scatterplot of the RGB values of all the images.
        """
        mean_RGBs_and_red_ratios = np.load(self.data_path)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # for each cervix type, we plot a scatterplot of all the images' RGBs
        for typXRGBsRed, color, labl in zip(mean_RGBs_and_red_ratios,
                                            darkBrewerColors(),
                                            self.alltypes):
            # Extract the RGB values of each image
            allRGBs = np.array([RGBred[0] for RGBred in typXRGBsRed])
            reds = allRGBs[:, 0]
            greens = allRGBs[:, 1]
            blues = allRGBs[:, 2]
            ax.scatter(reds, greens, blues, c=color, label=labl)

        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")
        ax.set_title("RGB-space scatterplot of images\n")
        plt.legend()
        plt.show()

    def plot_percentage_red_pixels(self):
        """
        Plots the mean percentage of red pixels in each image, for each cervix
        type.
        """
        mean_RGBs_and_red_ratios = np.load(self.data_path)

        mean_red_percentage_bytype = [
                           100 * np.mean([RGBred[1] for RGBred in typXRGBsRed])
                           for typXRGBsRed in mean_RGBs_and_red_ratios]

        df_to_plot = pd.DataFrame([mean_red_percentage_bytype],
                                  columns=self.alltypes)
        fig = df_to_plot.plot.bar()
        fig.set_xticks([])
        fig.set_ylabel("Percentage")
        fig.set_title("Mean percentage of red pixels in image")
        plt.show(fig)
