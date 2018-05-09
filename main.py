from PIL import Image
import numpy as np
import os

from parser import ImageNetParser
from dataset import Dataset
from models import linear_model

def main():
    home_dir = os.environ['HOME']
    root_dir = os.path.join(home_dir, "resources/datasets/tiny-imagenet-200/tiny-imagenet-200") #path to imgnet root directory
    word_filename = os.path.join(root_dir, "words.txt")
    train_filename = os.path.join(root_dir, "wnids.txt")
    validation_filename = os.path.join(root_dir, "val/val_annotations.txt")

    print(root_dir)
    print(word_filename)
    print(train_filename)
    print(validation_filename)

    parser = ImageNetParser()
    dataset = parser.create_dataset(train_filename, validation_filename, word_filename, root_dir)

    print("trainingset size: %d" % len(dataset.training_images))
    print("validationset size: %d" % len(dataset.validation_images))

    model = linear_model()
    model.summary()

    train_x = dataset.get_training_data()
    train_y = dataset.get_training_labels_onehot()
    test_x = dataset.get_validation_data()
    test_y = dataset.get_validation_labels_onehot()

    print(train_x.shape)
    print(train_x[0].shape)

    print(train_y.shape)
    print(train_y[0].shape)

    print(test_x.shape)
    print(test_x[0].shape)

    print(test_y.shape)
    print(test_y[0].shape)

    results = model.fit(
        train_x, train_y,
        epochs=2,
        batch_size=500,
        validation_data=(test_x, test_y)
    )

    return 0

if __name__ == "__main__":
    main()