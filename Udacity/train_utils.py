import argparse

from keras.engine.saving import load_model
from keras.preprocessing import image
from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
import math
import keras.backend as K
import cv2
import numpy as np
import matplotlib.image as mpimg

from Udacity.model import load_data, s2b
from Udacity.utils import batch_generator


def draw_image_with_label(img, label, prediction=None):
    theta = label * 0.69  # Steering range for the car is +- 40 degrees -> 0.69 radians
    line_length = 50
    line_thickness = 3
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
        int((img.shape[1] / 2) + (line_length * math.sin(theta))), int(img.shape[0] - (line_length * math.cos(theta))))
        image_draw.line([first_point, second_point], fill=prediction_line_color, width=line_thickness)

    del image_draw
    #plt.imshow(draw_image)
    #plt.show()
    return draw_image

def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, range_x, range_y, p_x, p_y):
    """
    Randomly shift the image virtially and horizontally (translation).
    """
    # trans_x = range_x * (np.random.rand() - 0.5)
    # trans_y = range_y * (np.random.rand() - 0.5)
    trans_x = range_x * p_x
    trans_y = range_y * p_y
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image


def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    IMAGE_WIDTH = image.shape[1]
    IMAGE_HEIGHT = image.shape[0]
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set current1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line:
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(image, steering_angle, range_x=100, range_y=10):
    """
    Generate an augumented image and adjust steering angle.
    (The steering angle is associated with the center image)
    """
    # image, steering_angle = random_flip(image, steering_angle)
    image = random_translate(image, range_x, range_y)
    # image = random_shadow(image)
    # image = random_brightness(image)
    return image, steering_angle


def print_graph(args, X_train, X_valid, y_train, y_valid):
    gen = batch_generator(args.data_dir, X_train, y_train, args.batch_size, True)
    for i in range(10):
        images, steers = gen.__next__()

        im = np.array(images[20], dtype=np.uint8)
        im = cv2.resize(im, None, fx=3, fy=3)
        im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        cv2.imshow('current1', im)
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # cv2.imshow('hi', im)
        cv2.waitKey(2000)

        hist = np.histogram(a=steers, bins=25)
        plt.figure(i)
        mean = np.mean(hist[1])
        plt.title(str(mean))
        plt.plot(hist[1][:-1], hist[0])
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.3)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=10)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=25000)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=80)
    parser.add_argument('-o', help='save best models only', dest='save_best_only', type=s2b, default='true')
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    parser.add_argument('-c', help='center only', dest='center_only', type=int, default=0)
    args = parser.parse_args()

    pc = r"S:\Netanel\Carnet\Record\IMG\center_2018_11_09_15_06_51_950.JPG"
    pl = r"S:\Netanel\Carnet\Record\IMG\left_2018_11_09_15_06_51_950.JPG"
    pr = r"S:\Netanel\Carnet\Record\IMG\right_2018_11_09_15_06_51_950.JPG"
    '''
    save_dir = r'D:\pres_imgs'
    im = cv2.imread(pc, current1)
    iml = cv2.imread(pl, current1)
    imr = cv2.imread(pr, current1)
    # im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    height, width = im.shape[:2]

    param = 30
    pts1 = np.float32([[param, param], [height-param, param], [0, width], [height, width]])
    pts2 = np.float32([[0, 0], [height, 0], [0, width], [height, width]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst_right = cv2.warpPerspective(imr, M, (width, height))

    image_ag_1 = random_translate(iml, 100, 10, -0.5, 0)
    image_ag_2 = random_translate(imr, 100, 10, 0.5, 0)
    cv2.imwrite(path.join(save_dir, 'left.JPG'), image_ag_1)
    cv2.imwrite(path.join(save_dir, 'right.JPG'), image_ag_2)
    cv2.imwrite(path.join(save_dir, 'right_eitan.JPG'), dst_right)
    # cv2.imshow("orig", im)
    # cv2.imshow("dsad", dst)
    # cv2.imshow("aug_1", image_ag_1)
    # cv2.imshow("aug_2", image_ag_2)
    # cv2.waitKey(0)
    # draw_image_with_label(im,0.5,-0.5)
    '''
    data = load_data(args)
    print_graph(args, *data)
