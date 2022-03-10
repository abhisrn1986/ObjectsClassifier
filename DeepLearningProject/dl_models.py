from email.mime import application
import enum
import logging
import os.path

from cv2 import COLOR_BGR2RGB, cvtColor
from keras.models import load_model
from tensorflow import keras
# model for NN composed of stack of layers connected sequentially (Feed Forward Network)
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from tensorflow import keras


from tensorflow.keras.preprocessing.image import (
    img_to_array,
    load_img,
    save_img)

import numpy as np

from image_importer import reshape_img_data





def get_ann_model(x_data, y_data, retrain = False, model_file_path = None):# define the model

    if retrain or not os.path.exists(model_file_path):
        K.clear_session()
        model = Sequential([
                            Dense(units=500, activation=keras.activations.relu,input_shape= x_data[0].shape), # 10 neurons in the hidden layer is arbitrary
                            Dropout(0.5),
                            Dense(units=4, activation=keras.activations.softmax) # 10 neurons in the output layer because we have 10 classes /digits
        ])

        callback = EarlyStopping(monitor='val_loss', patience=3)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.3), loss=keras.losses.categorical_crossentropy, metrics=keras.metrics.Precision())
        model.fit(x_data,y_data, epochs = 30, batch_size = 32, verbose = 1, callbacks = [callback], validation_split=0.2)

        model.save(model_file_path)
    else :

        model = load_model(model_file_path)


    return model


def get_cnn_model(x_data, y_data, retrain = False, model_file_path = None):
    if retrain or not os.path.exists(model_file_path):

        nRows,nCols,nDims = x_data.shape[1:]

        x_data = x_data.reshape(x_data.shape[0], nRows, nCols, nDims)
        input_shape = (nRows, nCols, nDims)

        model = Sequential()
        # The first two layers with 32 filters of window size 3x3
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        callback = EarlyStopping(monitor='val_loss', patience=3)

        model.fit(x_data,y_data, epochs = 30, batch_size = 32, verbose = 1, callbacks = [callback], validation_split=0.2)

        model.save(model_file_path)
    else :
        model = load_model(model_file_path)

    return model


def classify_webcam_image(model, image, classes):

    image = cvtColor(image, COLOR_BGR2RGB)

    logging.debug(f'captured image shape {image.shape}')


    image = image.reshape(1,224,224,3)

    # image = reshape_img_data(image)

    logging.debug(f'Processed image shape {image.shape}')

    prediction = model.predict(image)

    max_prob = -1.0
    max_index = -1
    for i, prob in enumerate(prediction[0]) :
        if prob > max_prob:
            max_index = i
            max_prob = prob

    return f'{classes[max_index]} : {max_prob}'


def classify_webcam_image_vgg16(model, image):


    image_batch = np.expand_dims(np.array(image), axis=0)
    image_batch.shape

    processed_image = keras.applications.vgg16.preprocess_input(image_batch)
    prediction = model.predict(processed_image)
    decoded_prediction = keras.applications.imagenet_utils.decode_predictions(prediction)
    return f'{decoded_prediction[0][0][1]} : {decoded_prediction[0][0][2]}'


def get_vgg16_model() :

    return keras.applications.vgg16.VGG16()



