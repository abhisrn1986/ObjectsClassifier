import os
import numpy as np

from tensorflow.keras.preprocessing.image import (
    img_to_array,
    load_img,
    save_img)


def get_image_data(base_path, class_folders):
    x_data = []
    y_data = []

    for target in class_folders:
        class_dir = base_path+target+'/'
        files = os.listdir(class_dir)
        for i_file, file in enumerate(files):
            file_path = f'{class_dir}{file}'
            # load the image
            img = load_img(path=file_path)  # target_size=(224, 224)
            # convert it to an array
            img_array = img_to_array(img)
            # append the array to X
            x_data.append(img_array)
            # append the numeric target to y
            y_data.append(target)

            # TODO remove this after testing all the images are imported properly
            # save_img('/home/abhishek/spiced_projects/'
            #           'ordinal-oregano-student-code/'
            #           'DeepLearningProject/data/test/'
            #           f'{target}{i_file}.png',
            #          x=np.array(img_array))

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    return x_data, y_data

    # TODO add this later on when needed
    # shuffle the data
    # shuffler = np.random.permutation(len(X))
    # X = X[shuffler]
    # y = y[shuffler]
