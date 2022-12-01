import os


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def join_paths(*paths):
    return os.path.normpath(os.path.sep.join(path.rstrip(r"\/") for path in paths))
