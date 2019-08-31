from keras.models import *
from keras.layers import *
from defines import ACTIVATION, LAYERS_NUM


def build_model(lr, dimension, channels=3, pretrained_weights=None):
    filters = 64
    kernel_size = 3  # Highly important to keep image size the same through layer
    strides = 1  # Highly important to keep image size the same through layer
    padding = "same"  # Highly important to keep image size the same through layer
    inp = Input(shape=(dimension, dimension, channels))

    z = (Conv2D(
        filters=channels,
        kernel_size=kernel_size,
        activation="relu",
        padding=padding,
        strides=strides,
        input_shape=(None, None, channels)
    ))(inp)  # layer 1
    # Create inner Conv Layers
    for layer in range(LAYERS_NUM):
        z = (Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=ACTIVATION))(
            z)

    z = (Conv2D(filters=channels, kernel_size=kernel_size, strides=strides, padding=padding, activation="linear"))(
        z)  # 8 - last layer - no relu

    # Residual layer
    out = add([z, inp])
    # FCN Model with residual connection
    model = Model(inputs=inp, outputs=out)
    model.compile(loss='mae', optimizer='adam')

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
