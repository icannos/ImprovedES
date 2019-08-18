"""
author: Maxime Darrin
"""

import os

# Used to enforce the use of the CPU (since we have different processes, which use a keras model, we cannot share the
# cpu
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Our Evoluation Strategy library
import ImprovedES as IES

from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D

import cv2

import numpy as np



class smallDino():
    @staticmethod
    def init_parameters():
        """
        This function returns a randomly
        :return:
        """
        # These shapes are extracted manually from the keras model in order to reduce the construction cost
        shapes = [(8, 8, 1, 16), (16,), (4, 4, 16, 16), (16,), (3, 3, 16, 16),
                  (16,), (2800, 256), (256,), (256, 3), (3,)]

        weights = []
        for s in shapes:
            weights.append(np.random.uniform(-2, 2, s))
        return np.asarray(weights)

    @staticmethod
    def distance(theta1, theta2):
        """
        Computes the "genetic" distance between two specimens"
        :param theta1:
        :param theta2:
        :return:
        """
        n = len(theta1)
        d = 0
        for i in range(n):

            d += np.sum((theta1[i] - theta2[i])**2) / n

        return d

    def model_from_parameters(self, theta):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='same',
                         input_shape=(50, 200, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.compile(loss='mse', optimizer="adam")

        model.set_weights(theta)

        return model

    @staticmethod
    def display_model_shape():
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='same',
                         input_shape=(50, 200, 1)))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (4, 4), strides=(2, 2), padding='same'))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dense(3))
        model.compile(loss='mse', optimizer="adam")

        shapes = []
        for w in model.get_weights():
            shapes.append(w.shape)

        print(shapes)



class dinoModelStructure(IES.ModelStructure):
    """
    Convolutionnal model for the chrome dino
    """

    @staticmethod
    def init_parameters():
        """
        This function returns a randomly
        :return:
        """

        # These shapes are extracted manually from the keras model in order to reduce the construction cost
        shapes = [(5, 5, 3, 8), (8,), (3, 3, 8, 8), (8,), (171072, 16), (16,), (16, 3), (3,)]
        weights = []
        for s in shapes:
            weights.append(np.random.uniform(-10, 10, s))
        return np.asarray(weights)

    @staticmethod
    def distance(theta1, theta2):
        """
        Computes the "genetic" distance between two specimens"
        :param theta1:
        :param theta2:
        :return:
        """
        n = len(theta1)
        d = 0
        for i in range(n):
            d += np.sum(theta1[i] - theta2[i]) / n

        return d

    @staticmethod
    def model_from_parameters(theta):
        model = Sequential()
        model.add(Conv2D(8, kernel_size=(5, 5),
                         activation='relu',
                         input_shape=(150, 600, 3)))
        model.add(Conv2D(8, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(16, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.set_weights(theta)

        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])

        return model


class dinoModelStructureDense(IES.ModelStructure):
    """
    This is a model for the chrome dino, it uses only Dense Layers. It is quite simple but I do think it is more
    appropriate thant convolutionnal network.
    """

    @staticmethod
    def init_parameters():

        # These shapes are extracted manually from the keras model in order to reduce the construction cost
        shapes = [(270000, 128), (128,), (128, 16), (16,), (16, 3), (3,)]
        weights = []
        for s in shapes:
            weights.append(np.random.uniform(-10, 10, s))
        return np.asarray(weights)

    @staticmethod
    def distance(theta1, theta2):
        n = len(theta1)
        d = 0
        for i in range(n):
            d += np.sum(theta1[i] - theta2[i]) / n

        return d

    @staticmethod
    def model_from_parameters(theta):
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(150 * 600 * 3,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.set_weights(theta)

        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])

        return model

    @staticmethod
    def display_model_shape():
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape=(150 * 600 * 3,)))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(3, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer="adam",
                      metrics=['accuracy'])

        shapes = []
        for w in model.get_weights():
            shapes.append(w.shape)

        print(shapes)
