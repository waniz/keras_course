"""
http://online.cambridgecoding.com/notebooks/cca_admin/convolutional-neural-networks-with-keras
https://elitedatascience.com/keras-tutorial-deep-learning-in-python
http://machinelearningmastery.com/handwritten-digit-recognition-using-convolutional-neural-networks-python-keras/

MNIST dataset

"""


import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from matplotlib import pyplot as plt


np.random.seed(42)

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255




