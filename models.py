import numpy as np
from keras import models
from keras import layers

def linear_model(input_dims, nb_labels):
    model = models.Sequential()
    model.add(layers.Dense(10000, input_shape=input_dims))
    model.add(layers.Dense(5000, activation='relu'))
    model.add(layers.Dense(nb_labels, activation='relu'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def conv_model(input_dims, nb_labels):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(64, 64, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(32, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3)))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(layers.Dense(64))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nb_labels))
    model.add(layers.Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model
