# from asyncio.log import logger
import logging
logger = logging.getLogger('simpleExample')
import os
import numpy as np
import cv2
from tensorflow import keras
from dl_models import NNType

from tensorflow.keras.preprocessing.image import (
    img_to_array,
    load_img,
    save_img)

from tensorflow.keras.utils import to_categorical


def import_class_imgs(data_dir, classes, nn_type, batch_size = 128):

    if(nn_type == NNType.MOBLIE_NET):
        data_gen = keras.preprocessing.image.ImageDataGenerator(
        # define the preprocessing function that should be applied to all images
        preprocessing_function=keras.applications.mobilenet_v2.preprocess_input,

            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True, 
            zoom_range=0.2

        )
    elif(nn_type == NNType.VGG16):
        data_gen = keras.preprocessing.image.ImageDataGenerator(
        # define the preprocessing function that should be applied to all images
        preprocessing_function=keras.applications.vgg16.preprocess_input,

            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True, 
            zoom_range=0.2

        )
    elif(nn_type == NNType.RES_NET):
        data_gen = keras.preprocessing.image.ImageDataGenerator(
        # define the preprocessing function that should be applied to all images
        preprocessing_function=keras.applications.resnet_v2.preprocess_input,

            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True, 
            zoom_range=0.2

        )

    train_data_gen = data_gen.flow_from_directory(
        directory=data_dir,
        class_mode="categorical",
        classes=classes,
        batch_size=batch_size,
        target_size=(224, 224)
    )


    # load in all images at once
    x_train, y_train = next(train_data_gen)

    return x_train, y_train


# TODO remove this afterwards. Old way of getting the images.

# def reshape_img_data(x_data):
#     x_data = np.array(x_data)
#     if(x_data.ndim == 4) : #color image
#         x_data = x_data.reshape(x_data.shape[0], x_data.shape[1] * x_data.shape[2] * x_data.shape[3])
#     else: # grayscale image
#         x_data = x_data.reshape(x_data.shape[0], x_data.shape[1] * x_data.shape[2])
    
#     return x_data

# def get_img_data(filepath) :
#     image = load_img(filepath)
#     data = [img_to_array(image)]
#     return reshape_img_data(data)


# def get_image_data(base_path, class_folders):
#     x_data = []
#     y_data = []

#     for i_target, target in enumerate(class_folders):
#         class_dir = base_path+target+'/'
#         files = os.listdir(class_dir)
#         for i_file, file in enumerate(files):
#             file_path = f'{class_dir}{file}'
#             # load the image
#             img = load_img(path=file_path)  # target_size=(224, 224)
#             # convert it to an array
#             img_array = img_to_array(img)
#             # append the array to X
#             x_data.append(img_array)
#             # append the numeric target to y
#             y_data.append(i_target)

#             # TODO remove this after testing all the images are imported properly
#             # save_img('/home/abhishek/spiced_projects/'
#             #           'ordinal-oregano-student-code/'
#             #           'DeepLearningProject/data/test/'
#             #           f'{target}{i_file}.png',
#             #          x=np.array(img_array))


#     x_data = np.array(x_data)
#     print("Shape of image", x_data.shape)

#     # x_data = reshape_img_data(x_data)
#     y_data = np.array(y_data)
#     y_data = to_categorical(y_data)

#     return x_data, y_data

#     # TODO add this later on when needed
#     # shuffle the data
#     # shuffler = np.random.permutation(len(X))
#     # X = X[shuffler]
    # y = y[shuffler]