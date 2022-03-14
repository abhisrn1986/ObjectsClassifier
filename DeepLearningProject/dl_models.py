import logging
logger = logging.getLogger('simpleExample')
import os.path
from enum import Enum

import numpy as np
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import load_model
from tensorflow import keras
# model for NN composed of stack of layers connected sequentially (Feed Forward Network)
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential


class NNType(Enum):
    CNN = 1
    VGG16 = 2
    MOBILE_NET_V2 = 3
    RES_NET_V2 = 4


def get_model_filename(model, base_path):
    
    filename = ""
    for layer in model.layers:
        filename += f'{layer.name}_{layer.output_shape[1]}_' 

    model.summary(print_fn = logger.debug)
    return base_path + filename[:-1] + '.h5'


def fit_model(model, x_data, y_data) :
    callback = EarlyStopping(monitor='val_loss', patience=2)
    return model.fit(x_data, y_data, 
            epochs=50, 
            verbose=2,
            batch_size=len(x_data), 
            callbacks=[callback],
            # use 30% of the data for validation
            validation_split=0.3).history



def get_cnn_model(x_data):

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

    return model



def get_mobile_net_model(n_classes):

    base_model = keras.applications.mobilenet_v2.MobileNetV2(
    weights='imagenet', 
    alpha=1.4,         # specific parameter of this model, small alpha reduces the number of overall weights
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
    model.add(keras.layers.Dense(n_classes, activation='softmax'))
    # have a look at the trainable and non-trainable params statistic

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

    return model


def get_res_net_model(n_classes):

    base_model = keras.applications.ResNet50V2(
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
    model.add(keras.layers.Dense(n_classes, activation='softmax'))
    # have a look at the trainable and non-trainable params statistic

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

    return model


def get_vgg16_model(n_classes) :

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
    model.add(keras.layers.Dense(n_classes, activation='softmax'))
    # have a look at the trainable and non-trainable params statistic

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])

    return model


def get_nn_model(x_data, y_data, nn_type, classes, retrain = False, models_dir = None):

    n_classes = len(classes)
    hist = None

    if(NNType.CNN == nn_type):
        model = get_cnn_model(x_data)
    if(NNType.VGG16== nn_type):
        model = get_vgg16_model(n_classes)
    if(NNType.MOBILE_NET_V2 == nn_type):
        model = get_mobile_net_model(n_classes)
    if(NNType.RES_NET_V2 == nn_type):
        model = get_res_net_model(n_classes)
    model_file_path = get_model_filename(model, models_dir)

    if retrain or not os.path.exists(model_file_path):
        hist = fit_model(model, x_data, y_data)
        model.save(model_file_path)
    else :
        logger.debug(f"Loaded model file {model_file_path}")
        model = load_model(model_file_path)

    return model, hist


def classify_webcam_image(model, image, classes, nn_type):

    pic_array = keras.preprocessing.image.img_to_array(image)
    image_batch = np.expand_dims(pic_array, axis=0)

    if(nn_type == NNType.MOBILE_NET_V2):
        processed_image = keras.applications.mobilenet_v2.preprocess_input(image_batch)

    elif(nn_type == NNType.VGG16):
        processed_image = keras.applications.vgg16.preprocess_input(image_batch)

    elif(nn_type == NNType.RES_NET_V2):

        processed_image = keras.applications.resnet_v2.preprocess_input(image_batch)
    else:
        processed_image = keras.applications.mobilenet_v2.preprocess_input(image_batch)


    prediction = model.predict(processed_image)

    max_prob = -1.0
    max_index = -1
    for i, prob in enumerate(prediction[0]) :
        if prob > max_prob:
            max_index = i
            max_prob = prob

    return prediction[0], classes[max_index], max_prob






