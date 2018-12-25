import numpy as np
import argparse
from os import path
from keras.preprocessing import image
from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
import math
import keras.backend as K
import cv2
import numpy as np

from model import load_data, s2b
from utils import batch_generator, translate_range_x, translate_range_y


def draw_image_with_label(img, label, prediction=None):
    #  theta = label * 0.69  # Steering range for the car is +- 40 degrees -> 0.69 radians
    theta = label * 1.5  # Steering range for the car is +- 40 degrees -> 0.69 radians
    line_length = 50
    line_thickness = 2
    label_line_color = (255, 0, 0)
    prediction_line_color = (0, 0, 255)
    pil_image = image.array_to_img(img, K.image_data_format(), scale=True)
    print('Actual Steering Angle = {0}'.format(label))
    draw_image = pil_image.copy()
    image_draw = ImageDraw.Draw(draw_image)
    first_point = (int(img.shape[1] / 2), img.shape[0])
    second_point = (
        int((img.shape[1] / 2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length * math.cos(theta))))
    image_draw.line([first_point, second_point], fill=label_line_color, width=line_thickness)

    if (prediction is not None):
        print('Predicted Steering Angle = {0}'.format(prediction))
        print('L1 Error: {0}'.format(abs(prediction - label)))
        theta = prediction * 4
        second_point = (
            int((img.shape[1] / 2) + (line_length * math.sin(theta))),
            int(img.shape[0] - (line_length * math.cos(theta))))
        image_draw.line([first_point, second_point], fill=prediction_line_color, width=line_thickness)

    del image_draw
    return draw_image

