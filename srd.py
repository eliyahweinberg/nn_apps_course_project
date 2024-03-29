import os
import cv2
import numpy as np
from utils import validate_dir

TRAIN_LR = 'train_lr'
TRAIN_SR = 'train_sr'
TEST_LR = 'test_lr'
TEST_SR = 'test_sr'


class SRDataset:
    def __init__(self, train_path, test_path, save_data_path, scaling_factor=2, target_size=128, is_padding=False, imgloader=cv2.imread):
        self.train_filenames = os.listdir(train_path)
        self.test_filenames = os.listdir(test_path)
        self.save_data_path = save_data_path
        validate_dir(self.save_data_path)
        self.train_path = train_path
        self.test_path = test_path
        self.target_size = target_size
        self.scaling_factor = scaling_factor
        self.is_padding = is_padding
        self.imgloader = imgloader

    def load_data(self, is_rebuild_train=False, is_rebuild_test=False):
        # try load saved sets
        print('Loading data')
        train_lr = self.load_set(os.path.join(self.save_data_path, TRAIN_LR))
        train_sr = self.load_set(os.path.join(self.save_data_path, TRAIN_SR))
        if (train_lr is None or train_sr is None) or is_rebuild_train:
            print('Creating set...')
            train_lr, train_sr = self._create_set(self.train_path, self.train_filenames)
            self.save_set(os.path.join(self.save_data_path, TRAIN_LR), train_lr)
            self.save_set(os.path.join(self.save_data_path, TRAIN_SR), train_sr)

        test_lr = self.get_test_lr()
        test_sr = self.get_test_sr()
        if (test_lr is None or test_sr is None) or is_rebuild_test:
            print('Creating set...')
            test_lr, test_sr = self._create_set(self.test_path, self.test_filenames)
            self.save_set(os.path.join(self.save_data_path, TEST_LR), test_lr)
            self.save_set(os.path.join(self.save_data_path, TEST_SR), test_sr)
        print('Data Loaded')
        return [train_lr, train_sr], [test_lr, test_sr]

    @staticmethod
    def save_set(name, data):
        np.save(name+'.npy', data)

    @staticmethod
    def load_set(name):
        try:
            data = np.load(name + '.npy')
        except FileNotFoundError:
            data = None
        return data

    def get_test_lr(self):
        return self.load_set(os.path.join(self.save_data_path, TEST_LR))

    def get_test_sr(self):
        return self.load_set(os.path.join(self.save_data_path, TEST_SR))

    def _create_set(self, path, files):
        lrs, srs = [], []

        for file in files:
            filename, extension = os.path.splitext(file)

            image = self.imgloader(os.path.join(path, filename + extension))
            if extension in ['.png', '.jpg']:
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image = np.stack((image,) * 3, axis=-1)
                if self.is_padding:
                    image = self._add_padding(image)
                image = image.astype('float32')
                # create LR image = input

                lr, sr = self.get_x_y(image, self.target_size, self.scaling_factor)
                lrs.append(lr)
                srs.append(sr)

        return np.asarray(lrs), np.asarray(srs)

    def _save_file(self, lr, sr, filename, lr_path, sr_path):
        for save_path in [lr_path, sr_path]:
            validate_dir(save_path)
        cv2.imwrite(os.path.join(lr_path, filename), lr)
        cv2.imwrite(os.path.join(sr_path, filename), sr)

    @staticmethod
    def _add_padding(image):
        old_size = image.shape[:2]  # old_size is in (height, width) format

        dimensions = max(old_size)
        # new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        # image = cv2.resize(image, (new_size[1], new_size[0]))

        delta_w = dimensions - old_size[1]
        delta_h = dimensions - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        return padded

    @staticmethod
    def get_x_y(image, target_size, scaling_factor):
        lr = cv2.resize(image, (target_size // scaling_factor, target_size // scaling_factor))

        lr = cv2.resize(lr, (target_size, target_size))

        # create SR image = label
        sr = cv2.resize(image, (target_size, target_size))
        return lr, sr
