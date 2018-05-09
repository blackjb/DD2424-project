import numpy as np
from keras import models
from keras import layers

def linear_model(input_dims, output_dims):
    model = models.Sequential()
    model.add(layers.Dense(10000, input_shape=(input_dims,)))
    model.add(layers.Dense(5000, activation='relu'))
    model.add(layers.Dense(output_dims, activation='relu'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model