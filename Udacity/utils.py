import cv2
import matplotlib.image as mpimg
import numpy as np
import os

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
augment_prob = 0.6

brightness_low = 0.8
brightness_high = 1.2

translate_multiplier = 0.003  # 0.002 in original
translate_range_x = 100
translate_range_y = 10

flip_prob = 0.5

steering_addition = 0.2


def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def preprocess(image):
    image = image[60:-25, :, :]  # crop
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    choice = np.random.choice(3)
    image = center

    if choice == 0:
        image = left
        steering_angle += steering_addition
    elif choice == 1:
        image = right
        steering_angle -= steering_addition

    return load_image(data_dir, image), steering_angle


def random_flip(image, steering_angle):
    if np.random.rand() < flip_prob:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def translate(image, steering_angle, trans_x, trans_y):
    steering_angle += trans_x * translate_multiplier
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_translate(image, steering_angle):
    trans_x = translate_range_x * np.random.uniform(-0.5, 0.5)
    trans_y = translate_range_y * np.random.uniform(-0.5, 0.5)
    return translate(image, steering_angle, trans_x, trans_y)


def random_shadow(image):
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
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
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = np.random.uniform(brightness_low, brightness_high)
    hsv[:, :, 2] = hsv[:, :, 2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def generate_shadow_coordinates(imshape, no_of_shadows=1):
    vertices_list = []
    for index in range(no_of_shadows):
        vertex = []
        for dimensions in range(np.random.randint(3, 15)):  ## Dimensionality of the shadow polygon
            vertex.append((imshape[1] * np.random.uniform(), imshape[0] // 3 + imshape[0] * np.random.uniform()))
        vertices = np.array([vertex], dtype=np.int32)  ## single shadow vertices
        vertices_list.append(vertices)
    return vertices_list  ## List of shadow vertices


def add_shadow(image, no_of_shadows=1):
    image_HLS = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)  ## Conversion to HLS
    mask = np.zeros_like(image)
    imshape = image.shape
    vertices_list = generate_shadow_coordinates(imshape, no_of_shadows)  # 3 getting list of shadow vertices
    for vertices in vertices_list:
        cv2.fillPoly(mask, vertices,
                     255)  ## adding all shadow polygons on empty mask, single 255 denotes only red channel

    image_HLS[:, :, 1][mask[:, :, 0] == 255] = image_HLS[:, :, 1][mask[:, :,
                                                                  0] == 255] * 0.5  ## if red channel is hot, image's "Lightness" channel's brightness is lowered
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  ## Conversion to RGB
    return image_RGB


def augument(data_dir, center, left, right, steering_angle):
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle)
    image = random_shadow(image)
    # image = add_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]
            # argumentation
            if is_training and np.random.rand() < augment_prob:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
