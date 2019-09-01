import os
import json
import numpy as np
import matplotlib.pyplot as plt


def validate_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class MultiplePlot:
    def __init__(self, size, _dimensions):
        self.ax = []
        self.images_num = 0
        self.fig = plt.figure(figsize=size)
        self.dimensions = _dimensions

    def add(self, image, title):
        self.images_num += 1
        self.ax.append(self.fig.add_subplot(self.dimensions[0], self.dimensions[1], self.images_num))
        self.ax[-1].set_title(title)
        plt.imshow(image)

    def show(self):
        plt.show()

    def save_fig(self, name):
        plt.savefig(name)
