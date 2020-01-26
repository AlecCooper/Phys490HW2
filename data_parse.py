# ------------------------------------------------------------------------------
#
# Phys 490--Winter 2020
# Assignment 2
# Alexander Cooper
#
# ------------------------------------------------------------------------------

import numpy as np
import random as rand

class Data():

    ## parses the data to be fed into our neural network
    ## file_location is the location of the csv file to be read
    ## n_test is the number of samples to set aside for the testing set
    ## x_len is the size of the each x vector, which is assumed to be 196 by default
    ## note: x data needs to be a uint8 and y data needs to be a ubyte
    def __init__(self,file_location,n_test,x_len=14*14):

        # import the data into a numpy array
        x_data = np.genfromtxt(file_location,usecols=range(x_len),dtype=np.uint8)
        y_data = np.genfromtxt(file_location,usecols=(x_len),dtype=np.ubyte)

        # Number of data points
        n_data = len(x_data)

        # test and train data
        x_train = np.empty((n_data-n_test,x_len),dtype=np.uint8)
        y_train = np.empty((n_data-n_test,1),dtype=np.ubyte)
        x_test = np.empty((n_test,x_len),dtype=np.uint8)
        y_test = np.empty((n_test,1),dtype=np.ubyte)

        # split into a training and testing set
        indicies = list(range(n_data))
        rand.shuffle(indicies)

        # randomly select our training and testing data
        train_index = indicies[0:n_data-n_test]
        test_index = indicies[-n_test:]

        # add our randomly selected data into the proper arrays
        n = 0
        for i in train_index:
            x_train[n] = x_data[i]
            y_train[n] = y_data[i]
            n+=1

        n = 0
        for i in test_index:
            x_test[n] = x_data[i]
            y_test[n] = y_data[i]
            n+=1

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test



