import os


def create_directory(path):
    """
    Create Directory of it already does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def join_paths(*paths):
    """
    Concatenate multiple paths.
    """
    return os.path.normpath(os.path.sep.join(path.rstrip(r"\/") for path in paths))
