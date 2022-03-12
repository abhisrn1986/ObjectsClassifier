import logging
import logging.config

logging.config.fileConfig('logging.conf')
# create logger
logger = logging.getLogger('global')
import os

import cv2
import numpy as np

from dl_models import (NNType, classify_webcam_image,
                       classify_webcam_image_vgg16, get_cnn_model,
                       get_nn_model, get_vgg16_model)
from image_importer import get_image_data, get_img_data, import_class_imgs
from utils import add_text, init_cam, key_action, write_image

dir_path = os.path.dirname(os.path.realpath(__file__))


g_nn_type = NNType.MOBLIE_NET
retrain = False
g_model_file_name = 'mobile_net.h5'

if __name__ == '__main__':

     # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

     # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None

    base_path = f'{dir_path}/data/images/'
    models_path = f'{dir_path}/data/models/'

    classes = ["fork", "book", "mug", "phone"]
    # x_train, y_train = get_image_data(base_path, classes)

    x_train, y_train = import_class_imgs(base_path, classes)

    logger.debug(f'Shape of the X {x_train.shape}')
    logger.debug(f'Shape of the y {y_train.shape}')



    model = get_nn_model(x_train, y_train, g_nn_type, classes, retrain,  models_path+g_model_file_name)

    
    # if (not g_use_vgg) :
    #     model = get_cnn_model(x_train, y_train,retrain, model_file_path = f'{models_path}cnn_model.h5')
    # else:
    #     model = get_vgg16_model()

    try:
        # q key not pressed 
        while key != 'q':
            # Capture frame-by-frame
            ret, frame = webcam.read()
            # fliping the image 
            frame = cv2.flip(frame, 1)
   
            # draw a [224x224] rectangle into the frame, leave some space for the black border 
            offset = 2
            width = 224
            x = 160
            y = 120
            cv2.rectangle(img=frame, 
                          pt1=(x-offset,y-offset), 
                          pt2=(x+width+offset, y+width+offset), 
                          color=(0, 0, 0), 
                          thickness=2
            )     
            
            # get key event
            key = key_action()
            
            # if key == 'space':
            # write the image without overlay
            # extract the [224x224] rectangle out of it
            image = frame[y:y+width, x:x+width, :]
            # write_image(out_folder, image) 
            # if (not g_use_vgg) :

            #     prediction = classify_webcam_image(model, image, classes)
            # else:
            #     prediction = classify_webcam_image_vgg16(model, image)

            prediction, category, max_prob = classify_webcam_image(model, image, classes)


            logging.debug(f'Prediction: {prediction}')
            # add_text(f'{np.round(prediction[0], decimals=2)}', frame)
            # add_text(f'{prediction}\n{category} : {max_prob}', frame)
            add_text(f'{category} : {max_prob}', frame)




            # disable ugly toolbar
            cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            cv2.imshow('frame', frame)            
            
    finally:
        # when everything done, release the capture
        logging.debug('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()


    # test_image = get_img_data('/home/abhishek/spiced_projects/'
    #                   'ordinal-oregano-student-code/'
    #                   'DeepLearningProject/data/test/Mug15.png')

    # logging.debug(f"Test image shape is {test_image.shape}")

    # y = model.predict(test_image)


    # logger.debug(f"Prediction is {y}")




