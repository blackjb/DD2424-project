import pickle
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Dataset:
    def __init__(self):
        self.training_images = []
        self.training_labels = []
        self.validation_images = []
        self.validation_labels = []
        self.labels = []
        self.label_names = []
        self.information = {}

    def load(self, filename):
        """
        Imports saved
        :param filename: Filepath to the file containing the dataset
        """
        ds_file = open(filename, 'w+')
        dataset = pickle.load(ds_file)
        self.training_images = dataset.training_images
        self.training_labels = dataset.training_labels
        self.validation_images = dataset.validation_images
        self.validation_labels = dataset.validation_labels
        self.labels = dataset.labels
        self.label_names = dataset.label_names
        self.information = dataset.information

    def save(self, filename):
        """
        :param filename: Filepath to the file where the dataset will be stored
        """
        ds_file = open(filename, 'w+')
        pickle.dump(self, ds_file)

    def get_training_data(self):
        return self.training_images

    def get_training_labels(self):
        return self.training_labels

    def get_training_labels_onehot(self):
        encoder = OneHotEncoder()
        onehot_labels = encoder.fit_transform(self.training_labels.reshape(-1, 1))
        return onehot_labels

    def get_validation_data(self):
        return self.validation_images

    def get_validation_labels(self):
        return self.validation_labels

    def get_validation_labels_onehot(self):
        encoder = OneHotEncoder()
        onehot_labels = encoder.fit_transform(self.validation_labels.reshape(-1, 1))
        return onehot_labels
