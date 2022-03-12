import logging
import logging.config

logging.config.fileConfig('logging.conf')
# create logger
logger = logging.getLogger('global')
import os

import cv2
import numpy as np

from dl_models import NNType, classify_webcam_image, get_nn_model
from image_importer import import_class_imgs
from utils import add_text, init_cam, key_action

dir_path = os.path.dirname(os.path.realpath(__file__))


g_nn_type = NNType.RES_NET
retrain = True
# g_model_file_name = 'cnn.h5'
# g_model_file_name = 'mobile_net.h5'
# g_model_file_name = 'mobile_net200.h5'
g_model_file_name = 'resnet50_v2.h5'


if __name__ == '__main__':

     # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

     # also try out this resolution: 640 x 360
    webcam = init_cam(640, 480)
    key = None

    base_path = f'{dir_path}/data/images/'
    models_path = f'{dir_path}/data/models/'

    classes = os.listdir(base_path)

    x_train, y_train = import_class_imgs(base_path, classes, g_nn_type)
    logger.debug(f'Shape of the X {x_train.shape}')
    logger.debug(f'Shape of the y {y_train.shape}')

    model = get_nn_model(x_train, y_train, g_nn_type, classes, retrain,  models_path+g_model_file_name)


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

            prediction, category, max_prob = classify_webcam_image(model, image, classes)

            logging.debug(f'Prediction: {prediction}')


            # format the text
            formatted_str = ''
            for icatg, catg in enumerate(classes) :
                prob = round(prediction[icatg], 2)
                formatted_str += f'{catg} : {prob}'

            add_text(formatted_str, frame, pos = (30,80), font_size=0.8, thickness=1)

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



