from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization, AvgPool2D
from keras.models import Sequential
from utils import INPUT_SHAPE


def build_parameterized_net(normalize=(127.5-1.0), pooling_type='max', add_dense=False, activation_conv='elu', activation_dense='elu'):
    model = Sequential()
    model.add(Lambda(lambda x: x / normalize, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation=activation_conv, padding='same'))
    if pooling_type == 'max':
        model.add(MaxPooling2D(2, 2))
    else:
        model.add(AvgPool2D(2, 2))
    model.add(Conv2D(36, (5, 5), activation=activation_conv, padding='same'))
    if pooling_type == 'max':
        model.add(MaxPooling2D(2, 2))
    else:
        model.add(AvgPool2D(2, 2))
    model.add(Conv2D(48, (5, 5), activation=activation_conv, padding='same'))
    if pooling_type == 'max':
        model.add(MaxPooling2D(2, 2))
    else:
        model.add(AvgPool2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation=activation_conv, padding='same'))
    model.add(Conv2D(64, (3, 3), activation=activation_conv, padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    if add_dense:
        model.add(Dense(200, activation=activation_dense))
    model.add(Dense(100, activation=activation_dense))
    model.add(Dense(50, activation=activation_dense))
    model.add(Dense(10, activation=activation_dense))
    model.add(Dense(1))
    model.summary()

    return model


def build_nvidia_model_pooling(args):
    """
    Modified NVIDIA model
    """
    act = 'elu'
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation=act, padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(36, (5, 5), activation=act, padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(48, (5, 5), activation=act, padding='same'))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3, 3), activation=act, padding='same'))
    model.add(Conv2D(64, (3, 3), activation=act, padding='same'))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation=act))
    model.add(Dense(50, activation=act))
    model.add(Dense(10, activation=act))
    model.add(Dense(1))
    model.summary()

    return model


def build_nvidia_model(args):
    """
    Modified NVIDIA model
    """
    act = 'elu'
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation=act, strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation=act, strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation=act, strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation=act))
    model.add(Conv2D(64, (3, 3), activation=act))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation=act))
    model.add(Dense(50, activation=act))
    model.add(Dense(10, activation=act))
    model.add(Dense(1))
    model.summary()

    return model

def build_nvidia_model_tanh(args):
    """
    Modified NVIDIA model
    """
    act1 = 'relu'
    act2 = 'tanh'
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))
    model.add(Conv2D(24, (5, 5), activation=act1, strides=(2, 2)))
    model.add(Conv2D(36, (5, 5), activation=act1, strides=(2, 2)))
    model.add(Conv2D(48, (5, 5), activation=act1, strides=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation=act1))
    model.add(Conv2D(64, (3, 3), activation=act1))
    model.add(Dropout(args.keep_prob))
    model.add(Flatten())
    model.add(Dense(100, activation=act2))
    model.add(Dense(50, activation=act2))
    model.add(Dense(10, activation=act2))
    model.add(Dense(1))
    # model.summary()

    return model