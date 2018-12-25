import argparse
import base64
import os
import shutil
from datetime import datetime
from io import BytesIO
from os import path

import cv2
import eventlet.wsgi
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from keras.models import load_model
import utils
from Utils.train_utils import draw_image_with_label
from utils import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS


def display_info(image_view, steering):
    tmp_image = np.asarray(draw_image_with_label(image_view, steering))
    # tmp_image = cv2.resize(tmp_image, None, fx=3, fy=3)
    return tmp_image


def get_video_properties(m_vidcap):
    (major_ver, minor_ver, subminor_ver) = cv2.__version__.split('.')
    if int(major_ver) < 3:
        fps = m_vidcap.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = m_vidcap.get(cv2.CAP_PROP_FPS)

    vid_width = m_vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    vid_height = m_vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    length = int(m_vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    return int(vid_width), int(vid_height), fps, length


def prepeare_input_image(original_img):
    original_img = original_img[:, 30:-30, :]
    shape = original_img.shape
    ratio = IMAGE_WIDTH / shape[1]
    res = cv2.resize(original_img, None, fx=ratio, fy=ratio)
    shape = res.shape
    vid_frame_pred = res[shape[0] - 66:, :, :]  # crop
    cv2.imshow('original', original_img)
    cv2.waitKey(1)
    res = cv2.cvtColor(vid_frame_pred, cv2.COLOR_RGB2BGR)
    return res
    # predict the steering angle for the image


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create predicted video from video')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'video_path',
        type=str,
        nargs='?',
        default='',
        help='Path to video.'
    )

    args = parser.parse_args()

    model = load_model(args.model)

    vidcap = cv2.VideoCapture(args.video_path)
    vid_width, vid_height, fps, length = get_video_properties(vidcap)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    save_name = r'test_video.avi'
    video = cv2.VideoWriter(save_name, fourcc, fps, (vid_width, vid_height))
    success = True
    while success:
        success, vid_frame = vidcap.read()
        #  BGR format

        if success:
            vid_frame_pred = prepeare_input_image(vid_frame)

            # predict the steering angle for the image
            prediction = model.predict(np.array([vid_frame_pred]), batch_size=1)[0]
            steering_angle = float(prediction[0])
            predicted_velocity = float(prediction[1])

            detailed_img = display_info(vid_frame, steering_angle)
            cv2.imshow('preview', detailed_img)
            cv2.waitKey(1)
            video.write(detailed_img)

    cv2.destroyAllWindows()
    video.release()
