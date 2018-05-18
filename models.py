import numpy as np
from keras import models
from keras import layers

def linear_model(input_dims, nb_labels):
    #input_dims = (input_dims[0]*input_dims[1]*input_dims[2], )
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_dims))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dense(nb_labels, activation='softmax'))
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def minimal(input_dims, nb_labels):
    model = models.Sequential()
    model.add(layers.Dense(input_dims), activation='relu')
    model.add(layers.Dense(nb_labels, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

def conv_model(input_dims, nb_labels):
    model = models.Sequential()

    model.add(layers.Convolution2D(32, 3, 3, activation='relu', input_shape=input_dims))
    model.add(layers.Convolution2D(32, 3, 3, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(nb_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model



def alexnet_model1(input_dims, nb_labels, activation, optimizer, norm=False):
    model = models.Sequential()

    model.add(layers.Convolution2D(48, 7, 3, input_shape=input_dims))
    if norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='valid'))

    model.add(layers.Convolution2D(128, 5, 1))
    if norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='valid'))

    model.add(layers.Convolution2D(192, 3, 1))

    model.add(layers.Convolution2D(192, 3, 1))

    model.add(layers.Convolution2D(128, 3, 1))
    model.add(layers.Activation(activation))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='valid'))

    #Fully conected end layers
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation=activation))
    model.add(layers.Dense(2048, activation=activation))
    model.add(layers.Dense(nb_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def alexnet_model2(input_dims, nb_labels, activation, optimizer, norm=False):
    model = models.Sequential()

    model.add(layers.Convolution2D(92, 7, 3, input_shape=input_dims))
    if norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='valid'))

    model.add(layers.Convolution2D(256, 5, 1))
    if norm:
        model.add(layers.BatchNormalization())
    model.add(layers.Activation(activation))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='valid'))

    model.add(layers.Convolution2D(384, 3, 1))

    model.add(layers.Convolution2D(384, 3, 1))

    model.add(layers.Convolution2D(384, 3, 1))
    model.add(layers.Activation(activation))
    model.add(layers.MaxPool2D(pool_size=(3, 3), strides=(2,2), padding='valid'))

    #Fully conected end layers
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation=activation))
    model.add(layers.Dense(4096, activation=activation))
    model.add(layers.Dense(nb_labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model
