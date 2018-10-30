from keras.models import load_model
import sys
import numpy as np
import glob
import os
import cv2

if ('../../PythonClient/' not in sys.path):
    sys.path.insert(0, '../../PythonClient/')
from AirSimClient import *

# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob(r'C:\Users\netanelgip\Documents\Carnet\Project\EndToEndLearningRawData\model\models/*.h5')
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model
    
print('Using model {0} for testing.'.format(MODEL_PATH))

model = load_model(MODEL_PATH)

client = CarClient()
client.confirmConnection()
client.enableApiControl(True)
car_controls = CarControls()
print('Connection established!')

car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0

image_buf = np.zeros((1, 59, 255, 3))
state_buf = np.zeros((1,4))

def get_image():
    '''
    Receive an image from the simulator
    '''
    image_response = client.simGetImages([ImageRequest(0, AirSimImageType.Scene, False, False)])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
    
    return image_rgba[76:135,0:255,0:3].astype(float)


while True:
    car_state = client.getCarState()
    
    if (car_state.speed < 5):
        car_controls.throttle = 1.0
    else:
        car_controls.throttle = 0.0
    
    image_buf[0] = get_image()
    img = np.array(image_buf[0], dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("test", img)
    cv2.waitKey(5)
    state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
    model_output = model.predict([image_buf, state_buf])
    car_controls.steering = round(0.5 * float(model_output[0][0]), 2)

    print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))
    
    client.setCarControls(car_controls)
