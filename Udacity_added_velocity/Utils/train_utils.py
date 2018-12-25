import math

import keras.backend as K
from PIL import ImageDraw
from keras.preprocessing import image


def draw_image_with_label(img, label, prediction=None):
    theta = label * 2  # Steering range for the car is +- 40 degrees -> 0.69 radians
    line_length = 125
    line_thickness = 5
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

    if prediction is not None:
        print('Predicted Steering Angle = {0}'.format(prediction))
        print('L1 Error: {0}'.format(abs(prediction - label)))
        theta = prediction * 4
        second_point = (
        int((img.shape[1] / 2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length * math.cos(theta))))
        image_draw.line([first_point, second_point], fill=prediction_line_color, width=line_thickness)

    del image_draw
    return draw_image
