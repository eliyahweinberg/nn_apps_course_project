from keras.applications.vgg16 import VGG16
from keras.models import *
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import numpy as np
import json
import cv2
from utils import NumpyEncoder
from defines import INITIAL_LRATE, DROP, LEARNING_RATE_CYCLES, NB_PAIRS, NB_SCALING_STEPS, NB_STEPS, EPOCHS, FIVE, X, Y,\
                    CROP_FLAG, FLIP_FLAG, CROP_SIZE, NOISE_FLAG, BATCH_SIZE, BLUR_LOW, BLUR_HIGH, SORT, SORT_ORDER,\
                    SR_FACTOR, NOISY_PIXELS_STD, SHUFFLE


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


def train_perceptual_loss(model, name, train_set, test_set, input_shape, epochs, batch, use_checkpoint=True):
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
        filepath = "PretrainedWeights/weights_" + name + "-{epoch:02d}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='auto')
        callbacks_list.append(checkpoint)

    history = full_model.fit(train_set[X], y_train_loss_model, batch_size=batch, epochs=epochs, shuffle=True,
                             validation_data=[test_set[X], y_test_loss_model], callbacks=callbacks_list)

    with open('history_{}.json'.format(name), 'w') as f:
        json.dump(history.history, f, cls=NumpyEncoder)

    return history


# Add noise to lr sons
""" 0 is the mean of the normal distribution you are choosing from
    NOISY_PIXELS_STD is the standard deviation of the normal distribution
    (row,col,ch) is the number of elements you get in array noise   """
def add_noise(image):
    row, col, ch = image.shape

    noise = np.random.normal(0, NOISY_PIXELS_STD, (row, col, ch))
    #Check image dtype before adding.
    noise = noise.astype('float32')
    # We clip negative values and set them to zero and set values over 255 to it.
    noisy = np.clip((image + noise), 0, 255)

    return noisy


def preprocess(image, scale_fact, scale_fact_inter, i):

    # scale down is sthe inverse of the intermediate scaling factor
    scale_down = 1 / scale_fact_inter
    # Create hr father by downscaling from the original image
    hr = cv2.resize(image, None, fx=scale_fact, fy=scale_fact, interpolation=cv2.INTER_CUBIC)
    # Crop the HR father to reduce computation cost and set the training independent from image size
    #print("hr before crop:", hr.shape[0], hr.shape[1])
    if CROP_FLAG:
        h_crop = w_crop = np.random.choice(CROP_SIZE)
        #print("h_crop, w_crop:", h_crop, w_crop)
        if (hr.shape[0] > h_crop):
            x0 = np.random.randint(0, hr.shape[0] - h_crop)
            h = h_crop
        else:
            x0 = 0
            h = hr.shape[0]
        if (hr.shape[1] > w_crop):
            x1 = np.random.randint(0, hr.shape[1] - w_crop)
            w = w_crop
        else:
            x1 = 0
            w = hr.shape[1]
        hr = hr[x0 : x0 + h, x1 : x1 + w]
        #print("hr body:", x0, x0 + h, x1, x1 + w)
        #print("hr shape:", hr.shape[0], hr.shape[1])


    if FLIP_FLAG:
        # flip
        """ TODO check if 4 is correct or if 8 is better.
        Maybe change to np functions, as in predict permutations."""

        # if np.random.choice(4):
        #     flip_type = np.random.choice([1, 0, -1])
        #     hr = cv2.flip(hr, flip_type)
        #     if np.random.choice(2):
        #         hr = cv2.transpose(hr)
        k = np.random.choice(8)
        print(k)
        hr = np.rot90(hr, k, axes=(0, 1))
        if (k > 3):
            hr = np.fliplr(hr)

        # lr = cv2.flip( lr, flip_type )

    # hr is cropped and flipped then copies as lr
    # Blur lr son
    lr = cv2.resize(hr, None, fx=scale_down, fy=scale_down, interpolation=cv2.INTER_CUBIC)
    # Upsample lr to the same size as hr
    lr = cv2.resize(lr, (hr.shape[1], hr.shape[0]), interpolation=cv2.INTER_CUBIC)

    # Add gaussian noise to the downsampled lr
    if NOISE_FLAG:
        lr = add_noise(lr)


    # Expand image dimension to 4D Tensors.
    lr = np.expand_dims(lr, axis=0)
    hr = np.expand_dims(hr, axis=0)

    """ For readability. This is an important step to make sure we send the
    LR images as inputs and the HR images as targets to the NN"""
    X = lr
    y = hr

    return X, y


def s_fact(image, NB_PAIRS, NB_SCALING_STEPS):
    BLUR_LOW_BIAS = 0.0
    scale_factors = np.empty(0)
    if image.shape[0] * image.shape[1] <= 50 * 50:
        BLUR_LOW_BIAS = 0.3
    for i in range(NB_SCALING_STEPS):
        temp = np.random.uniform(BLUR_LOW + BLUR_LOW_BIAS, BLUR_HIGH,
                                 int(NB_PAIRS / NB_SCALING_STEPS))  # Low = 0.4, High = 0.95
        if SORT:
            temp = np.sort(temp)
        if SORT_ORDER == 'D':
            temp = temp[::-1]
        scale_factors = np.append(scale_factors, temp, axis=0)
        scale_factors = np.around(scale_factors, decimals=5)
    scale_factors_pad = np.repeat(scale_factors[-1], abs(NB_PAIRS - len(scale_factors)))
    scale_factors = np.concatenate((scale_factors, scale_factors_pad), axis=0)

    # Intermediate SR_Factors
    intermidiate_SR_Factors = np.delete(np.linspace(1, SR_FACTOR, NB_SCALING_STEPS + 1), 0)
    intermidiate_SR_Factors = np.around(intermidiate_SR_Factors, decimals=3)

    lenpad = np.int(NB_PAIRS / NB_SCALING_STEPS)
    intermidiate_SR_Factors = np.repeat(intermidiate_SR_Factors, lenpad)

    pad = np.repeat(intermidiate_SR_Factors[-1], abs(len(intermidiate_SR_Factors) - max(len(scale_factors), NB_PAIRS)))
    intermidiate_SR_Factors = np.concatenate((intermidiate_SR_Factors, pad), axis=0)

    #	scale_factors = np.vstack((scale_factors,a))
    return scale_factors, intermidiate_SR_Factors


def image_generator(image, NB_PAIRS, BATCH_SIZE, NB_SCALING_STEPS):
    i = 0
    scale_fact, scale_fact_inter = s_fact(image, NB_PAIRS, NB_SCALING_STEPS)
    while True:
        X, y = preprocess(image, scale_fact[i] + np.round(np.random.normal(0.0, 0.03), decimals=3),
                          scale_fact_inter[i], i)

        i = i + 1

        yield [X, y]


def train_zssr(model, image):
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate]

    history = model.fit_generator(image_generator(image, NB_PAIRS, BATCH_SIZE, NB_SCALING_STEPS),
                                  steps_per_epoch=NB_STEPS,
                                  epochs=EPOCHS,
                                  shuffle=SHUFFLE,
                                  callbacks=callbacks_list,
                                  max_queue_size=32,
                                  verbose=1)
    return history
