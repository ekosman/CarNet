import argparse
import base64
import os
import shutil
from datetime import datetime
from io import BytesIO

import cv2
import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from keras.models import load_model

import utils
from Utils.train_utils import draw_image_with_label

sio = socketio.Server()
app = Flask(__name__)
model = None

K = 1  # throttle multiplier
B = 5  # throttle bias


def display_info(image_view, steering):
    image = cv2.cvtColor(image_view, cv2.COLOR_RGB2BGR)
    tmp_image = np.asarray(draw_image_with_label(image, steering))
    tmp_image = cv2.resize(tmp_image, None, fx=3, fy=3)
    cv2.imshow('win', tmp_image)
    cv2.waitKey(1)


@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
            
        try:
            image = np.asarray(image)        # from PIL image to numpy array
            image = utils.preprocess(image)  # apply the preprocessing

            # predict the steering angle for the image
            prediction = model.predict(np.array([image]), batch_size=1)[0]
            steering_angle = float(prediction[0])
            predicted_velocity = float(prediction[1])
            throttle = K * (predicted_velocity - speed) + B

            display_info(image, steering_angle)

            print('Steering = {}, Throttle = {}, Speed = {}'.format(steering_angle, throttle, speed))
            send_control(steering_angle, throttle)
        except Exception as e:
            print(e)
        
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    model = load_model(args.model)

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

