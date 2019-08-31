import os
import cv2
import numpy as np
from utils import validate_dir

TRAIN_LR = 'train_lr'
TRAIN_SR = 'train_sr'
TEST_LR = 'test_lr'
TEST_SR = 'test_sr'


class SRDataset:
    def __init__(self, train_path, test_path, save_data_path, image_size=128, imgloader=cv2.imread):
        self.train_filenames = os.listdir(train_path)
        self.test_filenames = os.listdir(test_path)
        self.save_data_path = save_data_path
        validate_dir(self.save_data_path)
        self.train_path = train_path
        self.test_path = test_path
        self.image_size = image_size
        self.imgloader = imgloader

    def load_data(self):
        # try load saved sets
        print('Loading data')
        train_lr = self.load_set(os.path.join(self.save_data_path, TRAIN_LR))
        train_sr = self.load_set(os.path.join(self.save_data_path, TRAIN_SR))
        if train_lr is None or train_sr is None:
            print('No saved train set, creating set...')
            train_lr, train_sr = self._create_set(self.train_path, self.train_filenames)
            self.save_set(os.path.join(self.save_data_path, TRAIN_LR), train_lr)
            self.save_set(os.path.join(self.save_data_path, TRAIN_SR), train_sr)

        test_lr = self.load_set(os.path.join(self.save_data_path, TEST_LR))
        test_sr = self.load_set(os.path.join(self.save_data_path, TEST_SR))
        if test_lr is None or test_sr is None:
            print('No saved test set, creating set...')
            test_lr, test_sr = self._create_set(self.test_path, self.test_filenames)
            self.save_set(os.path.join(self.save_data_path, TEST_LR), test_lr)
            self.save_set(os.path.join(self.save_data_path, TEST_SR), test_sr)
        print('Data Loaded')
        return (train_lr, train_sr), (test_lr, test_sr)

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
                image = image.astype('float32')
                # create LR image = input
                lr = cv2.resize(image, (64, 64))

                lr = self._add_padding(cv2.resize(lr, (128, 128)))

                # create SR image = label
                sr = self._add_padding(cv2.resize(image, (128, 128)))
                lrs.append(lr)
                srs.append(sr)

        return np.asarray(lrs), np.asarray(srs)

    def _save_file(self, lr, sr, filename, lr_path, sr_path):
        for save_path in [lr_path, sr_path]:
            validate_dir(save_path)
        cv2.imwrite(os.path.join(lr_path, filename), lr)
        cv2.imwrite(os.path.join(sr_path, filename), sr)
    
    def _add_padding(self, image):
        old_size = image.shape[:2]  # old_size is in (height, width) format

        ratio = float(self.image_size) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        # new_size should be in (width, height) format

        image = cv2.resize(image, (new_size[1], new_size[0]))

        delta_w = self.image_size - new_size[1]
        delta_h = self.image_size - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        padded = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)

        return padded