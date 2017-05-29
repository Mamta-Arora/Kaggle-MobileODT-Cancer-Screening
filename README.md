# Kaggle-MobileODT-Cervical-Cancer
Classification of cervix types for cervical cancer screening. The classification is made from images and uses convolutional neural networks.

All of the code necessary for implementing the exploratory data analysis, benchmark models, preprocessing, and neural networks is in the folders `modules` and `workflow_classes`. For actually running these python scripts following a logical work flow, from beginning to end, the notebook `FullWorkflow.ipynb` is used.

## Setting up necessary data

In order to run the main notebook `FullWorkflow.ipynb`, it is necessary to first acquire the training data. This is done from the Kaggle competition website: https://www.kaggle.com/c/intel-mobileodt-cervical-cancer-screening/data . It is only necessary to download the files `test.7z` and `train.7z`.

This data should be placed in a folder called `Data` which lies in the same directory as the jupyter notebook: the training data should be in the folder `./Data/train` and the testing data should be in a folder `./Data/test`.

## Required python packages

The code has been written in the Anaconda distribution of Python 2.7. Additionally, the packages `tqdm` and `TensorFlow` need to be installed; the installation instructions are found [here](https://anaconda.org/conda-forge/tqdm) and [here](https://www.tensorflow.org/install/).
