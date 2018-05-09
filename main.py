from PIL import Image
import numpy as np
from sklearn.utils import shuffle
import os

from parser import ImageNetParser
from dataset import Dataset
from models import linear_model

import tensorflow as tf

def main():
    # Remove tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Set dataset directory variables
    home_dir = os.environ['HOME']
    root_dir = os.path.join(home_dir, "resources/datasets/tiny-imagenet-200/tiny-imagenet-200") #path to imgnet root directory
    word_filename = os.path.join(root_dir, "words.txt")
    train_filename = os.path.join(root_dir, "wnids.txt")
    validation_filename = os.path.join(root_dir, "val/val_annotations.txt")

    # Parase images and create dataset
    parser = ImageNetParser()
    dataset = parser.create_dataset(train_filename, validation_filename, word_filename, root_dir)

    print("trainingset size: %d" % len(dataset.training_images))
    print("validationset size: %d" % len(dataset.validation_images))

    # Fetch training data
    train_x = dataset.get_training_data()
    train_y = dataset.get_training_labels_onehot()
    test_x = dataset.get_validation_data()
    test_y = dataset.get_validation_labels_onehot()

    # Shuffle training data
    train_x, train_y = shuffle(train_x, train_y, random_state=0)

    # Set input and label dimensionality variables
    input_dims = train_x.shape[1]
    output_dims = train_y.shape[1]

    # Assert correct dimensionality
    assert train_x.shape[0] == train_y.shape[0]
    assert test_x.shape[0] == test_y.shape[0]

    # Create model
    model = linear_model(input_dims, output_dims)
    model.summary()

    # Train model
    results = model.fit(
        train_x, train_y,
        epochs=2,
        batch_size=500,
        validation_data=(test_x, test_y)
    )

    return 0


if __name__ == "__main__":
    main()