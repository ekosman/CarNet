import argparse
import numpy as np
from matplotlib import pyplot as plt
from Udacity.model import load_data
from Udacity.utils import batch_generator


def print_graph(args, X_train, X_valid, y_train, y_valid):
    gen = batch_generator(args.data_dir, X_train, y_train, args.batch_size, True)
    for i in range(10):
        images, steers = gen.__next__()
        hist = np.histogram(a=steers, bins=25)
        plt.figure(i)
        mean = np.mean(hist[1])
        plt.title(str(mean))
        plt.plot(hist[1][:-1], hist[0])
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=20)
    args = parser.parse_args()

    data = load_data(args)
    print_graph(args, *data)
