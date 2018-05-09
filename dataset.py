import pickle
import numpy as np

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
        onehot_labels = [None for _ in range(len(self.training_labels))]
        for label in self.training_labels:
            onehot_vec = np.zeros(len(self.labels))
            onehot_vec[label] = 1
            onehot_vec = np.asarray(onehot_vec)
            onehot_labels[label] = onehot_vec
        labels = np.asarray(onehot_labels)
        return labels

    def get_validation_data(self):
        return self.validation_images

    def get_validation_labels(self):
        return self.validation_labels

    def get_validation_labels_onehot(self):
        onehot_labels = [None for _ in range(len(self.validation_labels))]
        for label in self.validation_labels:
            onehot_vec = np.zeros(len(self.labels))
            onehot_vec[label] = 1
            onehot_vec = np.asarray(onehot_vec)
            onehot_labels[label] = onehot_vec
        labels = np.asarray(onehot_labels)
        return labels
