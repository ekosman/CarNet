from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Lambda, Input, concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam, SGD, Adamax, Nadam


def initial_model(image_input_shape, state_input_shape):
    activation = 'relu'

    # Create the convolutional stacks
    pic_input = Input(shape=image_input_shape)

    img_stack = Conv2D(16, (3, 3), name="convolution0", padding='same', activation=activation)(pic_input)
    img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
    img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution1')(img_stack)
    img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
    img_stack = Conv2D(32, (3, 3), activation=activation, padding='same', name='convolution2')(img_stack)
    img_stack = MaxPooling2D(pool_size=(2, 2))(img_stack)
    img_stack = Flatten()(img_stack)
    img_stack = Dropout(0.2)(img_stack)

    # Inject the state input
    state_input = Input(shape=state_input_shape)
    merged = concatenate([img_stack, state_input])

    # Add a few dense layers to finish the model
    merged = Dense(64, activation=activation, name='dense0')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(10, activation=activation, name='dense2')(merged)
    merged = Dropout(0.2)(merged)
    merged = Dense(1, name='output')(merged)

    adam = Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model = Model(inputs=[pic_input, state_input], outputs=merged)
    model.compile(optimizer=adam, loss='mse')
    return model