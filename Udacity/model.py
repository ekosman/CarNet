import glob

import argparse as argparse
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
import itertools

np.random.seed(0)


def load_data(args):
    data_df = None
    dsa = args.data_dir
    X_train, X_valid, Y_train, Y_valid = None, None, None, None
    for filename in glob.iglob(path.join(args.data_dir, '**\\*.csv'), recursive=True):
        data_df = pd.read_csv(filename, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])
        X = data_df[['center', 'left', 'right']].values
        y = data_df['steering'].values

        if X_train is None:
            X_train, X_valid, Y_train, Y_valid = train_test_split(X, y, test_size=args.test_size, random_state=0)
            print('Loaded {} data items from: {}'.format(len(X_train) + len(X_valid), filename))
        else:
            X_train_tmp, X_valid_tmp, Y_train_tmp, Y_valid_tmp = train_test_split(X, y, test_size=args.test_size, random_state=0)
            X_train = np.concatenate((X_train, X_train_tmp), axis=0)
            X_valid = np.concatenate((X_valid, X_valid_tmp), axis=0)
            Y_train = np.concatenate((Y_train, Y_train_tmp), axis=0)
            Y_valid = np.concatenate((Y_valid, Y_valid_tmp), axis=0)
            print('Loaded {} data items from: {}'.format(len(X_train_tmp) + len(X_valid_tmp), filename))


    print('Data size: Total items - {}        Train - {}        Validation - {}'.format(len(X_train) + len(X_valid), len(X_train), len(X_valid)))
    return X_train, X_valid, Y_train, Y_valid


def train_model(save_dir, model, augment_prob, batch_size, small_angle_keep_prob, translate_multiplier, args, X_train, X_valid, y_train, y_valid):
    if not path.exists('D:\Eitan_Netanel\Models'):
        os.mkdir('D:\Eitan_Netanel\Models')

    full_path = path.join('D:\Eitan_Netanel\Models', save_dir)
    if not path.exists(full_path):
        os.mkdir(full_path)

    if not path.exists('centered_models') and args.center_only:
        os.mkdir('centered_models')

    elif not path.exists('augment_models') and not args.center_only:
        os.mkdir('augment_models')

    save_dir = full_path
    checkpoint = ModelCheckpoint(path.join(save_dir, 'model-{epoch:03d}.h5'),
                                 monitor='val_loss',
                                 verbose=0,
                                 mode='auto')
    early = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6,
                                  patience=10, verbose=1)

    if not path.exists('./logs'):
        os.mkdir('./logs')

    tensorboard = TensorBoard(log_dir='./logs')

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=args.learning_rate))

    model.fit_generator(batch_generator(args.data_dir, X_train, y_train, batch_size, True, augment_prob, small_angle_keep_prob, translate_multiplier),
                        steps_per_epoch=len(X_train) // batch_size,
                        epochs=args.nb_epoch,
                        validation_data=batch_generator(args.data_dir, X_valid, y_valid, batch_size, False, augment_prob, small_angle_keep_prob, translate_multiplier),
                        validation_steps=len(X_valid) // batch_size,
                        callbacks=[checkpoint, tensorboard, reduce_lr, early],
                        verbose=1)


def s2b(s):
    """
    Converts a string to boolean value
    """
    s = s.lower()
    return s == 'true' or s == 'yes' or s == 'y' or s == '1'


def train(normalize, pooling_type, add_dense, activation_conv, activation_dense, augment_prob, batch_size, small_angle_keep_prob, translate_multiplier, data, args):
    save_dir = 'normalize={}_pooling={}_addDence={}_activationConv={}_activationDence={}_augmentProb={}_batchSize={}_smallAngleKeepProb={}_translateMult={}'.format(normalize, pooling_type, add_dense, activation_conv, activation_dense, augment_prob, batch_size, small_angle_keep_prob, translate_multiplier)
    model = build_parameterized_net(normalize=normalize, pooling_type=pooling_type, add_dense=add_dense, activation_conv=activation_conv, activation_dense=activation_dense)
    train_model(save_dir, model, augment_prob, batch_size, small_angle_keep_prob, translate_multiplier, args, *data)


if __name__ == '__main__':
    """
        Load train/validation data set and train the model
    """
    parser = argparse.ArgumentParser(description='Behavioral Cloning Training Program')
    parser.add_argument('-d', help='data directory', dest='data_dir', type=str, default='data')
    parser.add_argument('-t', help='test size fraction', dest='test_size', type=float, default=0.2)
    parser.add_argument('-k', help='drop out probability', dest='keep_prob', type=float, default=0.5)
    parser.add_argument('-n', help='number of epochs', dest='nb_epoch', type=int, default=40)
    parser.add_argument('-s', help='samples per epoch', dest='samples_per_epoch', type=int, default=50000)
    parser.add_argument('-b', help='batch size', dest='batch_size', type=int, default=20)
    parser.add_argument('-l', help='learning rate', dest='learning_rate', type=float, default=5.0e-4)
    args = parser.parse_args()

    print('-' * 30)
    print('Parameters')
    print('-' * 30)
    for key, value in vars(args).items():
        print('{:<20} := {}'.format(key, value))
    print('-' * 30)

    augment_prob_value = np.arange(0.5, 0.6, 0.1)
    batch_size_values = [80]
    small_angle_keep_prob_values = np.arange(0.2, 0.3, 0.1)
    translate_mult_values = np.arange(0.003, 0.004, 0.001)
    normalize_values = [127.5]
    pooling_values = ['max', 'avg']
    add_dense_values = [False, True]
    activation_values_conv = ['elu', 'relu']
    activation_values_dense = ['tanh', 'elu']

    data = load_data(args)
    counter = 0
    for aug_prob, batch_size, angle_keep_prob, translate_mult, norm_value, pool_type, add_dense, act_conv, act_dense in itertools.product(augment_prob_value, batch_size_values, small_angle_keep_prob_values, translate_mult_values, normalize_values, pooling_values, add_dense_values, activation_values_conv, activation_values_dense):
        train(normalize=norm_value,
              pooling_type=pool_type,
              add_dense=add_dense,
              activation_conv=act_conv,
              activation_dense=act_dense,
              data=data,
              args=args,
              augment_prob=aug_prob,
              batch_size=batch_size,
              small_angle_keep_prob=angle_keep_prob,
              translate_multiplier=translate_mult)

        with open('counter.txt', 'w') as fp:
            fp.write(str(counter))

        counter += 1

