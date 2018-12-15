import argparse
from os import path

import numpy as np
from matplotlib import pyplot as plt
from Udacity.model import load_data
from Udacity.utils import batch_generator, random_brightness, random_flip, random_translate


def print_graph(args, X_train, X_valid, y_train, y_valid):
    gen = batch_generator(data_dir=args.data_dir,
                          image_paths=X_train,
                          steering_angles=y_train,
                          batch_size=args.batch_size,
                          is_training=True,
                          augment_prob=0.5,
                          small_angle_keep_prob=0.2,
                          translate_multiplier=0.003)

    for i in range(10):
        images, steers = gen.__next__()

        # im = np.array(images[20], dtype=np.uint8)
        # im = cv2.resize(im, None, fx=3, fy=3)
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # cv2.imshow('current1', im)
        # im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
        # cv2.imshow('hi', im)
        # cv2.waitKey(2000)

        hist = np.histogram(a=steers, bins=25)
        plt.figure()
        expectation = np.mean(steers)
        variance = np.var(steers)
        plt.title("Expectation: {}\nVariance: {}".format(expectation, variance))
        plt.plot(hist[1][:-1], hist[0])
        # plt.show()
        plt.savefig(path.join(r"S:\Netanel_Dolev\PycharmProjects\final_images", "augment_10000.png"))
        exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='D:\Eitan_Netanel\Records')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=10000)
    args = parser.parse_args()

    data = load_data(args)
    print_graph(args, *data)
