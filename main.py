import os
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
from parser import ImageNetParser
from dataset import Dataset
from models import linear_model, conv_model

def main():
    # Remove tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Set dataset directory variables
    home_dir = os.environ['HOME']
    root_dir = os.path.join(home_dir, "resources/datasets/tiny-imagenet-200/tiny-imagenet-200") #path to imgnet root directory
    word_filename = os.path.join(root_dir, "words.txt")
    train_filename = os.path.join(root_dir, "wnids.txt")
    validation_filename = os.path.join(root_dir, "val/val_annotations.txt")

    max_cat_size = 100

    # Parase images and create dataset
    parser = ImageNetParser()
    dataset = parser.create_dataset(train_filename, validation_filename, word_filename, root_dir, max_cat_size)

    print("trainingset size: %d" % len(dataset.training_images))
    print("validationset size: %d" % len(dataset.validation_images))

    # Fetch training data
    x_train = dataset.get_training_data()
    y_train = dataset.get_training_labels()
    x_test = dataset.get_validation_data()
    y_test = dataset.get_validation_labels()

    nb_labels = dataset.number_of_labels()




    # Get onehot label representation
    y_train = np_utils.to_categorical(y_train, num_classes=nb_labels)
    y_test = np_utils.to_categorical(y_test, num_classes=nb_labels)

    # Print data shape
    print("nb_labels: %d" % nb_labels)
    print("x_train.shape ", x_train.shape)
    print("y_train.shape ", y_train.shape)
    print("x_test.shape ", x_test.shape)
    print("y_test.shape ", y_test.shape)

    # Preprocessing of image data with datagenerators
    train_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False)
    train_generator.fit(x_train)

    validation_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=False)
    validation_generator.fit(x_test)

    # Set input and label dimensionality variables
    input_dims = x_train.shape[1:]
    nb_labels = y_train.shape[1]
    print("input_dims: ", input_dims)
    print("nb_labels: ", nb_labels)



    # Create model
    #model = linear_model(input_dims, nb_labels)
    model = conv_model(input_dims, nb_labels)
    model.summary()

    epochs = 1
    batch_size = 32
    # Train model
    model.fit_generator(train_generator.flow(x_train, y_train, batch_size=batch_size),
                        steps_per_epoch=len(x_train) / batch_size,
                        validation_data=(x_test, y_test),
                        epochs=epochs)

    eval = model.evaluate(x_test,
                   y_test,
                   batch_size=32)
    print(eval)
    return 0


if __name__ == "__main__":
    main()