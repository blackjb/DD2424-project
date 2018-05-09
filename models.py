import numpy as np
from keras import models
from keras import layers

def linear_model():
    model = models.Sequential()
    model.add(layers.Dense(1000, input_shape=(64*192,)))
    model.add(layers.Dense(200, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model