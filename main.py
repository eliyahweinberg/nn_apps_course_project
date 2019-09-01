from srd import SRDataset
import Models.unet as unet
import Models.zssr as zssr
from Models.train import train_perceptual_loss, train_zssr
from keras.layers import *
from utils import MultiplePlot
import os
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from defines import NB_CHANNELS, INITIAL_LRATE, WEIGHT_PASS, TRAIN_INPUT_DIR, TEST_INPUT_DIR, SAVED_SET_PATH,\
                    UNET_WEIGHTS, ZSSR_WEIGHTS, TEST_IMAGES_NUM, TEST_IMAGE_NAME, TEST_IMAGE_PATH


UNET = 0
ZSSR = 1


def create_dataset(dimensions):
    data_set = SRDataset(TRAIN_INPUT_DIR, TEST_INPUT_DIR, SAVED_SET_PATH)
    train_set, test_set = data_set.load_data(is_rebuild_train=True)

    for i in range(2):
        train_set[i] = np.reshape(train_set[i], (len(train_set[i]), dimensions, dimensions, NB_CHANNELS))
    for i in range(2):
        test_set[i] = np.reshape(test_set[i], (len(test_set[i]), dimensions, dimensions, NB_CHANNELS))

    return train_set, test_set


def create_model(model_type, dimensions, is_load_weights=True, is_weights_required=False):
    if model_type == UNET:
        model_handler = unet
        model_weights_name = UNET_WEIGHTS
    elif model_type == ZSSR:
        model_handler = zssr
        model_weights_name = ZSSR_WEIGHTS
    else:
        raise FileNotFoundError('Unsupported model')

    pretrained_weights = None
    if is_load_weights and os.path.isfile(WEIGHT_PASS + model_weights_name):
        pretrained_weights = WEIGHT_PASS + model_weights_name
    elif is_load_weights and is_weights_required:
        raise FileNotFoundError('Model weights [{}] not found'.format(pretrained_weights))

    model = model_handler.build_model(INITIAL_LRATE, dimensions, pretrained_weights=pretrained_weights)
    return model


def create_test_image(image_name, dimensions, scaling_factor):
    image = cv2.imread(image_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    lr, sr = SRDataset.get_x_y(image, target_size=dimensions, scaling_factor=scaling_factor)
    return lr, sr


def train_model(model, model_name, epochs, batch, dimensions, is_show_history=True):
    train_set, test_set = create_dataset(dimensions)
    history = train_perceptual_loss(model,
                                    model_name,
                                    train_set,
                                    test_set,
                                    (dimensions, dimensions, NB_CHANNELS),
                                    epochs,
                                    batch,
                                    use_checkpoint=False)
    model.save_weights(WEIGHT_PASS + '{}_trained_weights.h5'.format(model_name))

    if is_show_history:
        history_df = pd.DataFrame(history.history)
        history_df[['loss', 'val_loss']].plot()
        plt.show()
    return model, history


def process_zssr(image, low_res, target_size, scaling_factor, is_load_weights=False):
    image = cv2.resize(image, (target_size // scaling_factor, target_size // scaling_factor))
    model = create_model(ZSSR, None, is_load_weights=is_load_weights)
    history = train_zssr(model, image)
    low_res = np.expand_dims(low_res, axis=0)
    super_image = model.predict(low_res)
    return super_image


def train_unet_denoiser(epochs, batch, dimensions, is_show_history=True, is_load_weights=False):
    model = create_model(UNET, dimensions, is_load_weights=is_load_weights)
    model, history = train_model(model, 'unet', epochs, batch, dimensions, is_show_history=is_show_history)
    return model, history


def train_zssr_denoiser(epochs, batch, dimensions, is_show_history=True, is_load_weights=False):
    model = create_model(ZSSR, dimensions, is_load_weights=is_load_weights)
    model, history = train_model(model, 'zssr', epochs, batch, dimensions, is_show_history=is_show_history)
    return model, history


def denoise_image(image, model):

    image = image.astype('float32')
    # image = np.expand_dims(image, axis=0) already expanded by zssr
    super_image = model.predict(image)
    super_image = np.squeeze(super_image, axis=0)
    super_image = cv2.convertScaleAbs(super_image)
    return super_image


def test_model(denoiser_model, denoiser_name, image_dimensions, scaling_factor, image_name=None):
    if image_name is not None:
        test_images = [image_name]
    else:
        test_images = [os.path.join(TEST_IMAGE_PATH, TEST_IMAGE_NAME.format(i)) for i in range(1, TEST_IMAGES_NUM+1)]

    for image in test_images:
        plotter = MultiplePlot([20, 10], [2, 2])

        low_res, ground_truth = create_test_image(image, image_dimensions, scaling_factor)
        zssr_out = process_zssr(ground_truth, low_res, image_dimensions, scaling_factor)
        super_image = denoise_image(zssr_out, denoiser_model)

        zssr_out = np.squeeze(zssr_out, axis=0)
        zssr_out = cv2.convertScaleAbs(zssr_out)

        plotter.add(low_res, 'Low Resolution')
        plotter.add(ground_truth, 'Ground Truth')
        plotter.add(zssr_out, 'ZSSR Output')
        plotter.add(super_image, 'After {} denoise'.format(denoiser_name))
        if len(test_images) > 1:
            save_name = os.path.split(image)[-1]
        else:
            save_name = image
        plotter.save_fig('{}_{}'.format(denoiser_name, save_name))
        plotter.show()
        # cv2.imwrite('low_res.png', cv2.cvtColor(low_res, cv2.COLOR_RGB2BGR), params=[9])


def test_unet(image_dimensions=128, scaling_factor=2, image_name=None):
    model = create_model(UNET, image_dimensions, is_weights_required=True)
    test_model(model, 'unet', image_dimensions, scaling_factor, image_name)


def test_zssr(image_dimensions=128, scaling_factor=2, image_name=None):
    model = create_model(ZSSR, image_dimensions, is_weights_required=True)
    test_model(model, 'zssr_pretrained', image_dimensions, scaling_factor, image_name)


if __name__ == '__main__':
    test_unet()
    test_zssr()





