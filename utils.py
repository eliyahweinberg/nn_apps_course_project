import os
import json
import numpy as np


def validate_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)