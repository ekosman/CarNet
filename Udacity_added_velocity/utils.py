import cv2
import matplotlib.image as mpimg
import numpy as np
import os

IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 66, 200, 3
INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)
augment_prob = 0.7

brightness_low = 0.8
brightness_high = 1.2

translate_multiplier = 0.002
translate_range_x = 100
translate_range_y = 10

flip_prob = 0.5

steering_addition = 0.2

small_angle_keep_prob = 0.6
steering_threshold = 0.025


def load_image(data_dir, image_file):
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def preprocess(image):
    image = image[60:-25, :, :]  # crop
    image = cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)
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
    return cv2.flip(image, 1), -steering_angle


def translate(image, steering_angle, trans_x, trans_y, translate_multiplier=0.002):
    steering_angle += trans_x * translate_multiplier
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_translate(image, steering_angle, translate_multiplier):
    trans_x = translate_range_x * np.random.uniform(-0.5, 0.5)
    trans_y = translate_range_y * np.random.uniform(-0.5, 0.5)
    return translate(image, steering_angle, trans_x, trans_y, translate_multiplier)


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
        cv2.fillPoly(mask, vertices, 255)

    image_HLS[:, :, 1][mask[:, :, 0] == 255] = image_HLS[:, :, 1][mask[:, :, 0] == 255] * 0.5
    image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)  ## Conversion to RGB
    return image_RGB


def augment(data_dir, center, left, right, steering_angle, augment_prob, translate_multiplier):
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    if np.random.rand() < 0.5:
        image, steering_angle = random_flip(image, steering_angle)
    if np.random.rand() < augment_prob:
        image, steering_angle = random_translate(image, steering_angle, translate_multiplier)
    if np.random.rand() < augment_prob:
        image = add_shadow(image, np.random.choice(3) + 1)
    if np.random.rand() < augment_prob:
        image = random_brightness(image)
    return image, steering_angle


def batch_generator(data_dir, image_paths, steering_angles_and_velocity, batch_size, is_training, augment_prob, small_angle_keep_prob, translate_multiplier):
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers_velocity = np.empty((batch_size, 2))
    permutation = np.random.permutation(image_paths.shape[0])
    perm_i = 0
    while True:
        i = 0
        while i < batch_size:
            center, left, right = image_paths[perm_i]
            steering_angle = steering_angles_and_velocity[perm_i][0]

            # augmentation
            if is_training:
                image, steering_angle = augment(data_dir, center, left, right, steering_angle, augment_prob, translate_multiplier)
            else:
                image = load_image(data_dir, center)

            if np.abs(steering_angle) < steering_threshold and np.random.rand() > small_angle_keep_prob:
                continue

            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers_velocity[i][0] = steering_angle
            steers_velocity[i][1] = steering_angles_and_velocity[perm_i][1]   # Copy the velocity value and normalize it since tanh normalized
            i += 1
            perm_i +=1
            perm_i %= len(permutation)

        yield images, steers_velocity
