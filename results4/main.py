import os
import time

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

from parser import ImageNetParser
import models

def main():
    # Remove tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

	# Set pyplot backend to remove erro rwhen running over ssh
    plt.switch_backend('agg')
	

    # Set dataset directory variables
    home_dir = os.environ['HOME']
    root_dir = os.path.join(home_dir, "resources/datasets/tiny-imagenet-200") #path to imgnet root directory
    word_filename = os.path.join(root_dir, "words.txt")
    train_filename = os.path.join(root_dir, "wnids.txt")
    validation_filename = os.path.join(root_dir, "val/val_annotations.txt")

    max_cat_size = 500

    # Parase images and create dataset
    parser = ImageNetParser()
    dataset = parser.create_dataset(train_filename, validation_filename, word_filename, root_dir, max_cat_size)

    print("trainingset size: %d" % len(dataset.training_images))
    print("validationset size: %d" % len(dataset.validation_images))

    # Fetch training data
    x_train = dataset.get_training_data()
    x_train = x_train.astype('float32')/255.0
    y_train = dataset.get_training_labels()
    x_test = dataset.get_validation_data()
    x_test = x_test.astype('float32')/255.0
    y_test = dataset.get_validation_labels()

    # Set input and label dimensionality variables
    input_dims = x_train.shape[1:]
    nb_labels = dataset.number_of_labels()
    print("input_dims: ", input_dims)
    print("nb_labels: ", nb_labels)

    dataset = None # TODO remove? trying to clear memory


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
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False)
    train_generator.fit(x_train)

    validation_generator = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rotation_range=0,
        width_shift_range=0,
        height_shift_range=0,
        horizontal_flip=False)
    validation_generator.fit(x_train)

    # Create model
    model_list = [
                    #models.linear_model(input_dims, nb_labels),
                    models.alexnet_model2(input_dims, nb_labels, 'relu', 'adam'),
                    #models.alexnet_model1(input_dims, nb_labels, 'relu', 'adam', norm=True),
                    #models.alexnet_model1(input_dims, nb_labels, 'sigmoid', 'adam'),
                    #models.alexnet_model1(input_dims, nb_labels, 'relu', 'rmsprop'),
                    models.alexnet_model2(input_dims, nb_labels, 'relu', 'sgd')
                    ]

    # Set training hyper-parameteres
    epochs = 50
    batch_size = 64

    i = 1
    for model in model_list:
        start = time.time()

        model.summary()
        # Train model
        training_history = model.fit_generator(
                            train_generator.flow(x_train, y_train, batch_size=batch_size),
                            steps_per_epoch=len(x_train) / batch_size,
                            validation_data=validation_generator.flow(x_test, y_test, batch_size=batch_size),
                            epochs=epochs,
                            verbose=2
                            )

        print("training history:")
        print(training_history.history)

        pred = model.evaluate_generator(train_generator.flow(x_test, y_test, batch_size=batch_size))

        done = time.time()
        elapsed = done - start
        print("Training time: ")
        print(elapsed)

        plt.plot(range(epochs), training_history.history['val_loss'], 'r',
                 range(epochs), training_history.history['loss'], 'b')
        plt.savefig('loss-'+str(i)+'.png')
        plt.gcf().clear()
        plt.clf()

        plt.plot(range(epochs), training_history.history['val_acc'], 'r',
                 range(epochs), training_history.history['acc'], 'b')
        plt.savefig('acc-'+str(i)+'.png')
        plt.gcf().clear()
        plt.clf()
        i += 1

        # Evaluate the model
        print("Evaluation results: ", pred)



    return 0


if __name__ == "__main__":
    main()
