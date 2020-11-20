"""
Sample data can be seen in report
"""
from keras.models import load_model
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.preprocessing import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
import tensorflow as tf
import dataPreprocessing
import os

###################################### Main train and test Function ####################################################
"""
Main train and test function. You could call the subroutines from the `Subroutines` sections. Please kindly follow the 
train and test guideline.

`train` function should contain operations including loading training images, computing features, constructing models, training 
the models, computing accuracy, saving the trained model, etc
`test` function should contain operations including loading test images, loading pre-trained model, doing the test, 
computing accuracy, etc.
"""

def train(train_data_dir, model_dir, **kwargs):
    """Main training model.

    Arguments:
        train_data_dir (str):   The directory of training data
        model_dir (str):        The directory of the saved model.
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        train_accuracy (float): The training accuracy.
    """
    # we need a directory to store our augmented data
    save_dir = "./augmented/"

    # Data Preprocessing and Filtering
    dataPreprocessing.remove_ambiguous_image(data_dir = train_data_dir)
    dataPreprocessing.data_augmentation(train_dir=train_data_dir, save_dir = save_dir)

    # Set training and testing images
    train_dataset = image_dataset_from_directory(save_dir, batch_size=32, image_size=(150, 150),
                                                 color_mode='grayscale', seed=42, label_mode='categorical',)

    # validation_dataset = image_dataset_from_directory("./data/test/", batch_size=32, image_size=(150, 150),
    #                                                   color_mode='grayscale', seed=42, label_mode='categorical',)

    # Cache to Dataset
    print(train_dataset.class_names)

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    # validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    # Standardize
    normalization_layer = layers.experimental.preprocessing.Rescaling(1. / 255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    # validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Building model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(15, activation='softmax'))

    # Compilation using categorical crossentropy
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.summary()

    # Training the model
    # results = model.fit(train_dataset.repeat(), epochs=60, steps_per_epoch=30, validation_data=validation_dataset)
    results = model.fit(train_dataset.repeat(), epochs=60, steps_per_epoch=30)

    # Saving model
    model.save(os.path.join(model_dir,"my_classify_model.h5"))

    # Visualization of accuracies
    plt.plot(results.history['accuracy'], label="accuracy")
    # plt.plot(results.history['val_accuracy'], label="val_accuracy")
    plt.legend()
    plt.show()

    return results.history["accuracy"][-1]

def test(test_data_dir, model_dir, **kwargs):
    """Main testing model.

    Arguments:
        test_data_dir (str):    The `test_data_dir` is blind to you. But this directory will have the same folder structure as the `train_data_dir`.
                                You could reuse the snippets of loading data in `train` function
        model_dir (str):        The directory of the saved model. You should load your pretrained model for testing
        **kwargs (optional):    Other kwargs. Please specify default values if needed.

    Return:
        test_accuracy (float): The testing accuracy.
    """
    test_dataset = image_dataset_from_directory(test_data_dir, image_size=(150, 150), color_mode='grayscale',
                                                label_mode='categorical')
    model = load_model(os.path.join(model_dir,"my_classify_model.h5"))
    result = model.evaluate(test_dataset)
    print("loss = {}, accuracy = {}".format(str(result[0]), str(result[1])))
    return result[1]


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', default='train', choices=['train','test'])
    parser.add_argument('--train_data_dir', default='./data/train/', help='the directory of training data')
    parser.add_argument('--test_data_dir', default='./data/test/', help='the directory of testing data')
    parser.add_argument('--model_dir', default='model.pkl', help='the pre-trained model')
    opt = parser.parse_args()


    if opt.phase == 'train':
        training_accuracy = train(opt.train_data_dir, opt.model_dir)
        print(training_accuracy)

    elif opt.phase == 'test':
        testing_accuracy = test(opt.test_data_dir, opt.model_dir)
        print(testing_accuracy)
