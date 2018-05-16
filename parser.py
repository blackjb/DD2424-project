from dataset import Dataset
import os
from PIL import Image
import numpy as np



class Parser:
    """
    Abstract Parser class
    """
    def __init__(self, preprocessors=[]):
        self.preprocessors = preprocessors

    def create_dataset(self, file):
        """
        Imports image files runs image preprocessing and returns a Dataset object containing the imported images
        :return: dataset - Dataset object containing the imported iamges
        """
        raise NotImplementedError()

class ImageNetParser(Parser):
    """
    Parser for the image net dataset
    """
    def create_dataset(self, training_filename, validation_filename, word_filename, root_dir, max_cat_size):
        dataset = Dataset()
        train_file = open(training_filename)
        val_file = open(validation_filename)
        word_file = open(word_filename)

        labels = []
        label_names = []

        #import training images
        training_labels = []
        training_images = []
        for line in train_file:
            dir = os.path.join(root_dir, "train", line.strip(), "images")
            labels.append(line.strip())
            i = 0
            for image in os.listdir(dir):
                if image[0] == '.':
                    continue
                if i == max_cat_size:
                    break
                image_filename = os.path.join(dir, image)
                training_images.append(self.read_image(image_filename))
                training_labels.append(len(labels)-1)
                i += 1


        # Import validation images
        validation_labels = []
        validation_images = []
        val_dir = os.path.join(root_dir, "val/images")
        for line in val_file:
            line = line.split()
            label = line[1].strip()
            label = labels.index(label)
            validation_labels.append(label)
            filename = os.path.join(val_dir, line[0].strip())
            validation_images.append(self.read_image(filename))

        # Read label names
        words = {}
        for line in word_file:
            line = line.split()
            words[line[0].strip()] = line[1:]

        for label in labels:
            label_names.append(words[label])

        # Generate dataset information
        #TODO - add preprocessor information, training and validation size etc
        information = {}

        dataset.training_images = np.asarray(training_images)
        dataset.training_labels = np.asarray(training_labels)
        dataset.validation_images = np.asarray(validation_images)
        dataset.validation_labels = np.asarray(validation_labels)
        dataset.labels = labels
        dataset.label_names = label_names
        dataset.information = information

        return dataset

    def read_image(self, image_filename):
        """
        Reads an image file and
        :param image_filename:
        :return:
        """
        im = Image.open(image_filename)
        img = im.convert("RGB")

        #make img array
        img = np.asarray(img)
        im.close()

        return img
