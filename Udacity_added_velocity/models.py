from keras import Model
from keras.layers import Lambda, Conv2D, MaxPooling2D, Dropout, Dense, Flatten, AvgPool2D, concatenate
from keras.models import Sequential
from utils import INPUT_SHAPE


def get_velocity_branched_model(model):
    # Freeze all layers in the current model since we want to keep the steering as is
    for layer in model.layers:
        layer.trainable = False

    flat_output = model.get_layer('flatten_1').output
    steering_oputput = model.get_layer('dense_4').output
    velocity_branch = Dense(100, activation='relu', name='v_dense_1')(flat_output)
    velocity_branch = Dense(50, activation='relu', name='v_dense_2')(velocity_branch)
    velocity_branch = Dense(10, activation='relu', name='v_dense_3')(velocity_branch)
    velocity_branch = Dense(1, name='v_dense_4')(velocity_branch)
    merged = concatenate([steering_oputput, velocity_branch])
    model = Model(inputs=model.input, outputs=merged)
    model.summary()
    return model


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
