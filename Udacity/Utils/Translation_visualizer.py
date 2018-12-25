from utils import *
from os import path
from Utils.train_utils import *

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
