#Courtesy of https://towardsdatascience.com/writing-your-first-generative-adversarial-network-with-keras-2d16fd8d4889

from keras.datasets import mnist #Handwriting dataset
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam #Our optimization function

import matplotlib.pyplot as plt

import sys
import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 28 #rows for the image of MNIST
        self.img_cols = 28 #cols for the image of MNIST
        self.channels = 1   #Channels
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100 #Dimensions
    
        optimizer = Adam(0.0002, 0.5) #This is our optimizer!

        self.discriminator = self.build_discriminator()

        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.generator = self.build_generator()

        z = Input(shape = (self.latent_dim,))
        img = self.generator(z)

        self.discriminator.trainable = False

        validity = self.discriminator(img)

        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)


