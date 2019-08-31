from keras.models import *
from keras.layers import *
from keras.optimizers import *


def build_model(lr, dimension, channels=3, pretrained_weights=None):
    n1 = 64
    n2 = 128
    n3 = 256

    init = Input((dimension, dimension, channels))

    c1 = Conv2D(n1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(init)
    c1 = Conv2D(n1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1)

    x = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(n2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    c2 = Conv2D(n2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2)

    x = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(n3, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x = UpSampling2D()(c3)

    c2_2 = Conv2D(n2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    c2_2 = Conv2D(n2, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c2_2)

    m1 = Add()([c2, c2_2])
    m1 = UpSampling2D()(m1)

    c1_2 = Conv2D(n1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(m1)
    c1_2 = Conv2D(n1, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(c1_2)

    m2 = Add()([c1, c1_2])

    decoded = Conv2D(channels, 5,  activation='linear', padding='same')(m2)

    model = Model(init, decoded)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='mse', metrics=['accuracy'])

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model



### Deeper Unet ###
    # inputs = Input(input_size)
    # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    # drop4 = Dropout(0.5)(conv4)
    # pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    #
    # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    # drop5 = Dropout(0.5)(conv5)
    #
    # up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(drop5))
    # merge6 = concatenate([drop4, up6], axis=3)
    # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    #
    # up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(conv6))
    # merge7 = concatenate([conv3, up7], axis=3)
    # conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    #
    # up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(conv7))
    # merge8 = concatenate([conv2, up8], axis=3)
    # conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    #
    # up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #     UpSampling2D(size=(2, 2))(conv8))
    # merge9 = concatenate([conv1, up9], axis=3)
    # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    # conv10 = Conv2D(3, 1, activation='linear')(conv9)
    #
    # model = Model(input=inputs, output=conv10)
    #
    # model.compile(optimizer=Adam(lr=1e-4), loss='mse', metrics=['accuracy'])

    # model.summary()


