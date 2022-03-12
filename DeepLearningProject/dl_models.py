import logging
import os.path
from enum import Enum

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

class NNType(Enum):
    CNN = 1
    VGG16 = 2
    MOBLIE_NET = 3 


def get_cnn_model(x_data, y_data):

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
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    callback = EarlyStopping(monitor='val_loss', patience=3)

    model.fit(x_data,y_data, epochs = 30, batch_size = 128, verbose = 1, callbacks = [callback], validation_split=0.2)

    return model



def get_mobile_net_model(x_data, y_data, classes):

    base_model = keras.applications.mobilenet_v2.MobileNetV2(
    weights='imagenet', 
    alpha=0.35,         # specific parameter of this model, small alpha reduces the number of overall weights
    pooling='avg',      # applies global average pooling to the output of the last conv layer (like a flattening)
    include_top=False,  # we only want to have the base, not the final dense layers 
    input_shape=(224, 224, 3)
    )

    # freeze it!
    base_model.trainable = False

    model = keras.Sequential()
    model.add(base_model)
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(classes), activation='softmax'))
    # have a look at the trainable and non-trainable params statistic

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

    # observe the validation loss and stop when it does not improve after 3 iterations
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.fit(x_data, y_data, 
            epochs=50, 
            verbose=2,
            batch_size=len(x_data), 
            callbacks=[callback],
            # use 30% of the data for validation
            validation_split=0.3)

    return model


def get_vgg16_model(x_data, y_data, classes) :

    base_model = keras.applications.vgg16.VGG16(
    weights='imagenet', 
    pooling='avg',      # applies global average pooling to the output of the last conv layer (like a flattening)
    include_top=False,  # we only want to have the base, not the final dense layers 
    input_shape=(224, 224, 3)
    )

    # freeze it!
    base_model.trainable = False

    model = keras.Sequential()
    model.add(base_model)
    model.add(keras.layers.Dense(100, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(len(classes), activation='softmax'))
    # have a look at the trainable and non-trainable params statistic

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

    # observe the validation loss and stop when it does not improve after 3 iterations
    callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    model.fit(x_data, y_data, 
            epochs=50, 
            verbose=2,
            batch_size=len(x_data), 
            callbacks=[callback],
            # use 30% of the data for validation
            validation_split=0.3)

    return model


def get_nn_model(x_data, y_data, nn_type, classes, retrain = False, model_file_path = None):

    model = None

    if retrain or not os.path.exists(model_file_path):

        if(NNType.CNN == nn_type):
            model = get_cnn_model(x_data, y_data)
            model.save(model_file_path)
        if(NNType.VGG16== nn_type):
            model = get_vgg16_model(x_data, y_data, classes)
            model.save(model_file_path)
        if(NNType.MOBLIE_NET == nn_type):
            model = get_mobile_net_model(x_data, y_data, classes)
            model.save(model_file_path)
    else :
        model = load_model(model_file_path)

    return model


def classify_webcam_image(model, image, classes):

    pic_array = keras.preprocessing.image.img_to_array(image)
    image_batch = np.expand_dims(pic_array, axis=0)
    processed_image = keras.applications.mobilenet_v2.preprocess_input(image_batch)
    prediction = model.predict(processed_image)

    max_prob = -1.0
    max_index = -1
    for i, prob in enumerate(prediction[0]) :
        if prob > max_prob:
            max_index = i
            max_prob = prob

    return prediction[0], classes[max_index], max_prob






