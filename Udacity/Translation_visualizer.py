from utils import *
import cv2
from os import path
import numpy as np
from train_utils import *

data_dir = r"C:\Users\netanelgip\Documents\CarNet\Records\normal\IMG"
image_name = r"_2018_11_16_12_38_42_263.jpg"
save_dir = r'C:\Users\netanelgip\Documents\CarNet\Translate_visualizer'
save_name = path.join(*[save_dir, 'tmp' + '.avi'])
fps = 20

if __name__ == '__main__':
    # Open source images

    original_steering_angle = 0

    left_image = cv2.imread(path.join(data_dir, "left" + image_name), 1)
    center_image = cv2.imread(path.join(data_dir, "center" + image_name), 1)
    right_image = cv2.imread(path.join(data_dir, "right" + image_name), 1)
    vid_height, vid_width = center_image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print(save_name)
    video = cv2.VideoWriter(save_name, fourcc, fps, (vid_width-160, vid_height-85))
    #video = cv2.VideoWriter(save_name, fourcc, fps, (vid_width, vid_height))
    '''
    # translate left_image
    for trans_x in range(0,-81,-1):
        image,steering_angle = translate(left_image, 0, trans_x, 0)


        image = image[60:-25, 80:240, :]
        #image = left_image
        param = trans_x
        height = vid_height
        width = vid_width
        pts1 = np.float32([[param, param], [param, height - param], [width, height], [width, 0]])
        pts2 = np.float32([[0, 0], [0, height], [width, height], [width, 0]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (vid_width, vid_height))
        video.write(image)

    '''

    # translate left_image
    for trans_x in range(81, -81, -1):
        image, steering_angle = translate(left_image, original_steering_angle, trans_x, 0)
        image = image[60:-25, 80:240, :]
        image = np.asarray(draw_image_with_label(image, original_steering_angle, steering_angle))
        video.write(image)

    # translate center_image
    for trans_x in range(81, -81, -1):
        image, steering_angle = translate(center_image, original_steering_angle, trans_x, 0)
        image = image[60:-25, 80:240, :]
        image = np.asarray(draw_image_with_label(image, original_steering_angle, steering_angle))
        video.write(image)

        # translate right_image
    for trans_x in range(81, -81, -1):
        image, steering_angle = translate(right_image, original_steering_angle, trans_x, 0)
        image = image[60:-25, 80:240, :]
        image = np.asarray(draw_image_with_label(image, original_steering_angle, steering_angle))
        video.write(image)






    video.release()
