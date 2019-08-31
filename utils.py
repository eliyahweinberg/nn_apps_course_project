import os


def validate_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
