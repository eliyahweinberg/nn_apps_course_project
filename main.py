from srd import SRDataset
import Models.unet as unet
import Models.zssr as zssr
from Models.train import train_perceptual_loss
from keras.layers import *
from keras.optimizers import *
import os
import cv2
from defines import NB_CHANNELS, INITIAL_LRATE, WEIGHT_PASS


TRAIN_INPUT_DIR = 'Data/finished/train/dataraw/hires'
TEST_INPUT_DIR = 'Data/finished/valid/dataraw/hires'
SAVED_SET_PATH = 'Data/finished/set/'


def create_dataset(dimensions):
    data_set = SRDataset(TRAIN_INPUT_DIR, TEST_INPUT_DIR, SAVED_SET_PATH)
    train_set, test_set = data_set.load_data()

    for i in range(2):
        train_set[i] = np.reshape(train_set[i], (len(train_set[i]), dimensions, dimensions, NB_CHANNELS))
    for i in range(2):
        test_set[i] = np.reshape(test_set[i], (len(test_set[i]), dimensions, dimensions, NB_CHANNELS))

    return train_set, test_set


def main(is_train, epochs, batch, dimensions, model_weights_name=None, tf_gpu_session=False, tf_device_count={'GPU': 1, 'CPU': 4}):
    if tf_gpu_session:
        import tensorflow as tf
        config = tf.ConfigProto(device_count=tf_device_count)
        sess = tf.Session(config=config)
        K.set_session(sess)

    pretrained_weights = None
    if model_weights_name is not None and os.path.isfile(WEIGHT_PASS+model_weights_name):
        pretrained_weights = WEIGHT_PASS+model_weights_name

    pretrained_weights = WEIGHT_PASS + 'unet_weights.h5'
    unet_model = unet.build_model(INITIAL_LRATE, dimensions, pretrained_weights=pretrained_weights)
    pretrained_weights = WEIGHT_PASS + 'zssr_weights.h5'
    zssr_model = zssr.build_model(INITIAL_LRATE, dimensions, pretrained_weights=pretrained_weights)

    for i in range(5):
        if is_train:
            train_set, test_set = create_dataset(dimensions)
            train_perceptual_loss(unet_model, 'unet', train_set, test_set, (dimensions, dimensions, NB_CHANNELS), epochs, batch, i=str(i))
            unet_model.save_weights(WEIGHT_PASS + 'unet_weights_{}.h5'.format(i))

            train_perceptual_loss(zssr_model, 'zssr', train_set, test_set, (dimensions, dimensions, NB_CHANNELS), epochs, batch, i=str(i))
            zssr_model.save_weights(WEIGHT_PASS + 'zssr_weights_{}.h5'.format(i))

        data_set = SRDataset(TRAIN_INPUT_DIR, TEST_INPUT_DIR, SAVED_SET_PATH)
        ground_tr = data_set.get_test_sr()[2]
        image = data_set.get_test_lr()[2]
        image = np.expand_dims(image, axis=0)
        super_image_unet = unet_model.predict(image)
        super_image_zssr = zssr_model.predict(image)

        image = np.squeeze(image, axis=0)
        super_image_unet = np.squeeze(super_image_unet, axis=0)
        super_image_zssr = np.squeeze(super_image_zssr, axis=0)

        ground_tr = cv2.convertScaleAbs(ground_tr)
        image = cv2.convertScaleAbs(image)
        super_image_zssr = cv2.convertScaleAbs(super_image_zssr)
        super_image_unet = cv2.convertScaleAbs(super_image_unet)

        cv2.imwrite('low_res.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params=[9])
        cv2.imwrite('ground_truth.png', cv2.cvtColor(ground_tr, cv2.COLOR_RGB2BGR), params=[9])
        cv2.imwrite('super_res_unet__{}.png'.format(i), cv2.cvtColor(super_image_unet, cv2.COLOR_RGB2BGR), params=[9])
        cv2.imwrite('super_res_zssr__{}.png'.format(i), cv2.cvtColor(super_image_zssr, cv2.COLOR_RGB2BGR), params=[9])


    # history_df = pd.DataFrame(history.history)
    # history_df[['loss', 'val_loss']].plot()
    # history_df[['acc', 'val_acc']].plot()


    # plt.figure(figsize=(20, 4))
    # plt.imshow(sr)
    # plt.show()

if __name__ == '__main__':
    main(is_train=True,
         epochs=5,
         batch=16,
         dimensions=128,
         model_weights_name=None)

