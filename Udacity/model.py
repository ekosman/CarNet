import glob

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from utils import INPUT_SHAPE, batch_generator
import argparse
import os
from os import path
from models import *

np.random.seed(0)


def load_data(args):
    """
    Load training data and split it into training and validation set
    """
    # data_df = pd.read_csv(os.path.join(args.data_dir, 'driving_log.csv'))
    data_df = None
    dsa = args.data_dir
    X_train, X_valid, Y_train, Y_valid = None, None, None, None
    for filename in glob.iglob(path.join(args.data_dir, '**\\*.csv'), recursive=True):
        data_df = pd.read_csv(filename, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
        X = data_df[['center', 'left', 'right']].values
        y = data_df['steering'].values

        if X_train is None:
            X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
        else:
            X_train_tmp, X_valid_tmp, Y_train_tmp, Y_valid_tmp = train_test_split(X, y, test_size=args.test_size, random_state=0)
            X_train = np.concatenate((X_train, X_train_tmp), axis=0)
            X_valid = np.concatenate((X_valid, X_valid_tmp), axis=0)
            Y_train = np.concatenate((Y_train, Y_train_tmp), axis=0)
            Y_valid = np.concatenate((Y_valid, Y_valid_tmp), axis=0)

        print("X train size: {}".format(len(X_train)))


        # data_df = data_df.append(pd.read_csv(filename, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']), low_memory=False)
        # print(data_df.shape)
    # filename = r"D:\Eitan_Netanel\Records\normal\driving_log.csv"
    # data_df = pd.read_csv(filename, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
    # filename = r"D:\Eitan_Netanel\Records\normal_reverse\driving_log.csv"
    # data_df = data_df.append(pd.read_csv(filename, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']))



    # X = data_df[['center', 'left', 'right']].values
    # y = data_df['steering'].values
    #
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)

    return X_train, X_valid, Y_train, Y_valid


def train_model(model, args, X_train, X_valid, y_train, y_valid):
    """
    Train the model
    """
    if not path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not path.exists('centered_models') and args.center_only:
        os.mkdir('centered_models')

    elif not path.exists('augment_models') and not args.center_only:
        os.mkdir('augment_models')

    # save_dir = 'centered_models' if args.center_only else 'augment_models'
    save_dir = args.save_dir
    checkpoint = ModelCheckpoint(path.join(save_dir, 'model-{epoch:03d}.h5'),
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=args.save_best_only,
                                 mode='auto')
    early = EarlyStopping(monitor='val_loss', patience=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6,
                                  patience=5, verbose=1)

    if not path.exists('./logs'):
        os.mkdir('./logs')

    tensorboard = TensorBoard(log_dir='./logs')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, args.batch_size, True),
                        # steps_per_epoch=args.samples_per_epoch // args.batch_size,
                        steps_per_epoch=len(X_train) // args.batch_size,
                        epochs=args.nb_epoch,
                        max_q_size=1,
                        # validation_data=(X_valid, y_valid),
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, args.batch_size, False),
                        validation_steps=len(X_valid) // args.batch_size,
                        callbacks=[checkpoint, early, reduce_lr],
                        verbose=1)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def main():
    """
    Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory',        dest='data_dir',          type=str,   default='data')
    parser.add_argument('-t', help='test size fraction',    dest='test_size',         type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability',  dest='keep_prob',         type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs',      dest='nb_epoch',          type=int,   default=50)
    parser.add_argument('-s', help='samples per epoch',     dest='samples_per_epoch', type=int,   default=50000)
    parser.add_argument('-b', help='batch size',            dest='batch_size',        type=int,   default=20)
    parser.add_argument('-o', help='save best models only', dest='save_best_only',    type=s2b,   default='true')
    parser.add_argument('-l', help='learning rate',         dest='learning_rate',     type=float, default=1.0e-4)
    parser.add_argument('-c', help='center only',           dest='center_only',       type=int, default=0)
    parser.add_argument('-save_dir')
    parser.add_argument('-om', help='old model path')
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    if args.om is not None:
        model = load_model(args.om)

    else:
        model = build_nvidia_model(args)
    data = load_data(args)
    train_model(model, args, *data)


if __name__ == '__main__':
    main()

