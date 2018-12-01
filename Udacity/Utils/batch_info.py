import argparse

import cv2

import numpy as np

from Udacity.model import load_data
from Udacity.utils import batch_generator
from matplotlib import pyplot as plt


def print_graph(args, X_train, X_valid, y_train, y_valid):
    gen = batch_generator(args.data_dir, X_train, y_train, args.batch_size, True)
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
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=20)
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=1.0e-4)
    parser.add_argument('-c', help='center only', dest='center_only', type=int, default=0)
    args = parser.parse_args()

    data = load_data(args)
    print_graph(args, *data)
