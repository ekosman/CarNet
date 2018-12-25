import argparse
from os import path

import numpy as np
from matplotlib import pyplot as plt
from Udacity.model import load_data
from Udacity.utils import batch_generator, random_brightness, random_flip, random_translate

save_dir = r'S:\Netanel_Dolev\PycharmProjects\final_images\batch_samples'

def print_graph(args, X_train, X_valid, y_train, y_valid):
    gen = batch_generator(data_dir=args.data_dir,
                          image_paths=X_train,
                          steering_angles=y_train,
                          batch_size=args.batch_size,
                          is_training=True,
                          augment_prob=0.8,
                          small_angle_keep_prob=0.5,
                          translate_multiplier=0.003)

    images, steers = gen.__next__()

    hist = np.histogram(a=steers, bins=25)
    plt.figure()
    expectation = np.mean(steers)
    variance = np.var(steers)
    plt.title("Expectation: {}\nVariance: {}".format(expectation, variance))
    plt.plot(hist[1][:-1], hist[0])
    plt.show()
    plt.savefig(path.join(r"S:\Netanel_Dolev\PycharmProjects\final_images", "augment_10000.png"))
    plt.close()

    for i, img in enumerate(images):
        plt.figure()
        plt.imshow(np.array(img, dtype=np.uint8))
        plt.axis('off')
        plt.savefig(path.join(save_dir, '{}.png'.format(i)))
        plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='D:\Eitan_Netanel\Records')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=500)
    args = parser.parse_args()

    data = load_data(args)
    print_graph(args, *data)
