import os
import cv2
import numpy as np
from srd import SRDataset
from unet import unet
import matplotlib.pyplot as plt
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from zssr import step_decay
import json
import pandas as pd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

WEIGHT_PASS = 'model_weights/unet_weights_small_e3.h5'
TRAIN_INPUT_DIR = 'finished/train/dataraw/hires'
TEST_INPUT_DIR = 'finished/valid/dataraw/hires'
IMAGE_PNG = '.png'
TRAIN_DATA_DIR = 'finished/train/inputs'
TRAIN_LABELS_DIR = 'finished/train/labels'
SAVED_SET_PATH = 'finished/set/'

UNET_INPUT_HEIGHT = 512


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def train_model(model, name, train_lr, train_sr, test_lr, test_sr):
    lossModel = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    lossModel.trainable = False

    for layer in lossModel.layers:
        layer.trainable = False

    selectedLayers = [1, 2, 9, 10, 17, 18]

    # a list with the output tensors for each selected layer:
    selectedOutputs = [lossModel.layers[i].output for i in selectedLayers]
    # or [lossModel.get_layer(name).output for name in selectedLayers]

    # a new model that has multiple outputs:
    lossModel = Model(lossModel.inputs, selectedOutputs)
    lossOut = lossModel(model.output)

    fullModel = Model(model.input, lossOut)

    Y_train_lossModel = lossModel.predict(train_sr)
    Y_test_lossModel = lossModel.predict(test_sr)

    fullModel.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    fullModel.summary()
    lrate = LearningRateScheduler(step_decay)
    filepath = "model_weights/weights-improvement_"+name+"-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')

    callbacksList = [lrate, checkpoint]

    history = fullModel.fit(train_lr, Y_train_lossModel, batch_size=16, epochs=2, shuffle=True,
                            validation_data=[test_lr, Y_test_lossModel], callbacks=callbacksList)

    with open('history_{}.json'.format(name), 'w') as f:
        json.dump(history.history, f, cls=NumpyEncoder)


def main():
    pretrained_weights = None
    # # if os.path.isfile(WEIGHT_PASS):
    # #     pretrained_weights = WEIGHT_PASS
    data_set = SRDataset(TRAIN_INPUT_DIR, TEST_INPUT_DIR, SAVED_SET_PATH)
    (train_lr, train_sr), (test_lr, test_sr) = data_set.load_data()

    train_lr = np.reshape(train_lr, (len(train_lr), 128, 128, 3))
    train_sr = np.reshape(train_sr, (len(train_sr), 128, 128, 3))
    test_lr = np.reshape(test_lr, (len(test_lr), 128, 128, 3))
    test_sr = np.reshape(test_sr, (len(test_sr), 128, 128, 3))
    #
    unet_model = unet(pretrained_weights=pretrained_weights)

    # unet_model.fit(train_lr, train_sr,
    #                epochs=5,
    #                batch_size=16,
    #                shuffle=True,
    #                validation_data=(test_lr, test_sr))

    # unet_model.save_weights('model_weights/unet_weights_e4_32_10.h5')


    train_model(unet_model, 'unet', train_lr, train_sr, test_lr, test_sr)
    unet_model.save_weights('model_weights/unet_weights_100.h5')


    filters =  64
    NB_CHANNELS = 3
    LAYERS_NUM = 6
    ACTIVATION = 'relu'
    kernel_size = 3  # Highly important to keep image size the same through layer
    strides = 1  # Highly important to keep image size the same through layer
    padding = "same"  # Highly important to keep image size the same through layer
    inp = Input(shape=(128, 128, NB_CHANNELS))


    z = (Conv2D(
        filters=NB_CHANNELS,
        kernel_size=kernel_size,
        activation="relu",
        padding=padding,
        strides=strides,
        input_shape=(None, None, NB_CHANNELS)
    ))(inp)  # layer 1
    # Create inner Conv Layers
    for layer in range(LAYERS_NUM):
        z = (Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=ACTIVATION))(
            z)

    z = (Conv2D(filters=NB_CHANNELS, kernel_size=kernel_size, strides=strides, padding=padding, activation="linear"))(
        z)  # 8 - last layer - no relu

    # Residual layer
    out = add([z, inp])
    # FCN Model with residual connection
    zssr = Model(inputs=inp, outputs=out)
    # acc is not a good metric for this task*
    # compile model
    # Model summary
    # zssr.summary()
    # Plot model
    # from keras.utils import plot_model
    # plot_model(zssr, to_file = output_paths + 'zssr.png')

    train_model(zssr, 'zssr', train_lr, train_sr, test_lr, test_sr)
    zssr.save_weights('model_weights/zssr_weights_100.h5')


    # history_df = pd.DataFrame(history.history)
    # history_df[['loss', 'val_loss']].plot()
    # history_df[['acc', 'val_acc']].plot()

    # image = np.expand_dims(test_lr[2], axis=0)
    # super_image = model.predict(np.expand_dims(test_lr[2], axis=0))
    # # super_image = zssr.predict(image)
    # plt.figure(figsize=(20, 4))
    # # # plt.imshow(np.squeeze(image, axis=(0)))
    # # plt.imshow(get_Image(test_lr[1]))
    # #
    # # plt.show()
    # lr = cv2.convertScaleAbs(test_lr[2])
    # sr = cv2.convertScaleAbs(test_sr[2])
    # super_image = np.squeeze(super_image, axis=(0))
    # super_image = cv2.convertScaleAbs(super_image)
    #
    # cv2.imwrite('low_res.png', cv2.cvtColor(lr, cv2.COLOR_RGB2BGR), params=[9])
    # cv2.imwrite('ground_truth.png', cv2.cvtColor(sr, cv2.COLOR_RGB2BGR), params=[9])
    # cv2.imwrite('super_res.png', cv2.cvtColor(super_image, cv2.COLOR_RGB2BGR), params=[9])
    # plt.imshow(sr)
    # plt.show()

if __name__ == '__main__':
    main()

