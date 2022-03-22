import logging
import logging.config

logging.config.fileConfig('logging.conf')
# create logger
logger = logging.getLogger('simpleExample')
import os
import math
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import streamlit as st

from dl_models import NNType, classify_webcam_image, get_nn_model
from image_importer import import_class_imgs
from utils import add_text, init_cam, key_action

dir_path = os.path.dirname(os.path.realpath(__file__))

g_nn_type = NNType.RES_NET_V2
retrain = False
g_load_model_file = False

if __name__ == '__main__':

     # maybe you need this
    os.environ['KMP_DUPLICATE_LIB_OK']='True'

     # also try out this resolution: 640 x 360
    st.title("Webcam Live Feed")
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    webcam = init_cam(640, 480)
    key = None

    base_path = f'{dir_path}/data/images/'
    models_path = f'{dir_path}/data/models/'

    classes = os.listdir(base_path)

    x_train, y_train = import_class_imgs(base_path, classes, g_nn_type)
    logger.debug(f'Shape of the X {x_train.shape}')
    logger.debug(f'Shape of the y {y_train.shape}')


    if(g_load_model_file):
        model = load_model(models_path + "best_net.h5")
    else:
        model, hist = get_nn_model(x_train, y_train, g_nn_type, classes, retrain, models_path)


    try:
        # q key not pressed 
        while key != 'q' and run:
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

            prediction, category, max_prob = classify_webcam_image(model, image, classes, g_nn_type)

            logger.debug(f'Prediction: {prediction}')

            # format the text and add to the overlay
            formatted_str = ''
            for icatg, catg in enumerate(classes) :
                formatted_str += f'{catg[0]} : {prediction[icatg]:.2f} , '
            formatted_str = formatted_str[:-2]

            # map the color to red to green
            mapped_color = plt.cm.RdYlGn(max_prob)
            mapped_color = tuple(255 * x for x in mapped_color)
            mapped_color[:-1]
            add_text(formatted_str, frame, pos = (30,80), font_size=0.8, thickness=1)
            add_text(f'{category} : {max_prob}', frame, color = mapped_color)

            # disable ugly toolbar
            # cv2.namedWindow('frame', flags=cv2.WINDOW_GUI_NORMAL)              
            
            # display the resulting frame
            # cv2.imshow('frame', frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            FRAME_WINDOW.image(frame)         
            
    finally:
        # when everything done, release the capture
        logger.debug('quit webcam')
        webcam.release()
        cv2.destroyAllWindows()



