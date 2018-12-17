import os
from os import path

from keract import get_activations, display_activations
import cv2
from keras.engine.saving import load_model
import numpy as np
from Udacity import utils
from matplotlib import pyplot as plt

save_path = 'feature_maps'
img_path = r'D:\Eitan_Netanel\Records\Normal_3\IMG\center_2018_11_30_16_43_27_723.jpg'
model_path = r'C:\Users\netanelgip\PycharmProjects\CarNet\Udacity\normal_turns_reducelr_elu_lr0.6_batch20_saveall\model-016.h5'

# Load the image. the network expects RGB images
image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = utils.preprocess(image) # apply the preprocessing

model = load_model(model_path)
model.summary()
activations = get_activations(model, [image])

layer_names = list(activations.keys())
activation_maps = list(activations.values())
batch_size = activation_maps[0].shape[0]
assert batch_size == 1, 'One image at a time to visualize.'
for i, activation_map in enumerate(activation_maps):
    print('Displaying activation map {}'.format(i))
    shape = activation_map.shape

    layer_dir = path.join(save_path, layer_names[i]).replace('/', '').replace(':', '')
    if not path.exists(layer_dir):
        os.mkdir(layer_dir)

    if len(shape) == 4:
        # activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        edge_len = int(np.sqrt(shape[3]))
        activations = np.transpose(activation_map[0], (2, 0, 1))

        for j, act in enumerate(activations):
            plt.title(layer_names[i] + '_{}'.format(j))
            plt.imshow(act, interpolation='None', cmap='gray')
            # plt.show()

            save_name = path.join(layer_dir, str(j))
            print('saving to: {}'.format(save_name))
            plt.savefig(save_name)
            # activations = np.block(blocks)
    elif len(shape) == 2:
        # try to make it square as much as possible. we can skip some activations.
        activations = activation_map[0]
        num_activations = len(activations)
        if num_activations > 1024:  # too hard to display it on the screen.
            square_param = int(np.floor(np.sqrt(num_activations)))
            activations = activations[0: square_param * square_param]
            activations = np.reshape(activations, (square_param, square_param))
        else:
            activations = np.expand_dims(activations, axis=0)

        plt.title(layer_names[i])
        plt.imshow(activations, interpolation='None', cmap='gray')
        # plt.show()
        save_name = path.join(layer_dir, 'tmp')
        print('saving to: {}'.format(save_name))
        plt.savefig(save_name)
    else:
        raise Exception('len(shape) = 3 has not been implemented.')

