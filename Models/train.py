from keras.applications.vgg16 import VGG16
from keras.models import *
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import numpy as np
import json
from utils import NumpyEncoder
from defines import INITIAL_LRATE, DROP, LEARNING_RATE_CYCLES, NB_PAIRS, NB_SCALING_STEPS, NB_STEPS, EPOCHS, FIVE, X, Y


def step_decay(epochs):
    initial_lrate = INITIAL_LRATE
    drop = DROP
    if LEARNING_RATE_CYCLES:
        cycle = np.ceil(NB_PAIRS / NB_SCALING_STEPS)
        epochs_drop = np.ceil((NB_STEPS * EPOCHS) / NB_SCALING_STEPS)
        step_length = int(epochs_drop / FIVE)
    else:
        cycle = NB_PAIRS
        epochs_drop = np.ceil((NB_STEPS * EPOCHS) / FIVE)
        step_length = epochs_drop

    lrate = initial_lrate * np.power(drop, np.floor((1 + np.mod(epochs, cycle)) / step_length))
    print("lrate", lrate)
    return lrate


def train_perceptual_loss(model, name, train_set, test_set, input_shape, epochs, batch, use_checkpoint=True, i=''):
    loss_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    loss_model.trainable = False

    for layer in loss_model.layers:
        layer.trainable = False
    selected_layers = [1, 2, 9, 10, 17, 18]

    # a list with the output tensors for each selected layer:
    selected_outputs = [loss_model.layers[i].output for i in selected_layers]

    # a new model that has multiple outputs:
    loss_model = Model(loss_model.inputs, selected_outputs)
    loss_out = loss_model(model.output)

    full_model = Model(model.input, loss_out)

    y_train_loss_model = loss_model.predict(train_set[Y])
    y_test_loss_model = loss_model.predict(test_set[Y])

    full_model.compile(loss='mse', optimizer='adam', metrics=['accuracy', 'mse', 'mae'])
    full_model.summary()

    l_rate = LearningRateScheduler(step_decay)

    callbacks_list = [l_rate]
    if use_checkpoint:
        filepath = "PretrainedWeights/weights_" + name + "-{epoch:02d}__"+i+".hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
        callbacks_list.append(checkpoint)

    history = full_model.fit(train_set[X], y_train_loss_model, batch_size=batch, epochs=epochs, shuffle=True,
                             validation_data=[test_set[X], y_test_loss_model], callbacks=callbacks_list)

    with open('history_{}__{}.json'.format(name, i), 'w') as f:
        json.dump(history.history, f, cls=NumpyEncoder)

    return history
