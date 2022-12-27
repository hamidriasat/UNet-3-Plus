"""
General Utility functions
"""
import os
import tensorflow as tf


def create_directory(path):
    """
    Create Directory if it already does not exist.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def join_paths(*paths):
    """
    Concatenate multiple paths.
    """
    return os.path.normpath(os.path.sep.join(path.rstrip(r"\/") for path in paths))


def set_gpus(gpu_ids):
    """
    Change number of visible gpus for tensorflow.
    gpu_ids: Could be integer or list of integers.
    In case Integer: if integer value is -1 then use all available gpus.
    otherwise if positive number, then use given number of gpus.
    In case list of Integer: each integer will be considered as gpu id
    """
    all_gpus = tf.config.experimental.list_physical_devices('GPU')
    all_gpus_length = len(all_gpus)
    if isinstance(gpu_ids, int):
        if gpu_ids == -1:
            gpu_ids = range(all_gpus_length)
        else:
            gpu_ids = min(gpu_ids, all_gpus_length)
            gpu_ids = range(gpu_ids)

    selected_gpus = [all_gpus[gpu_id] for gpu_id in gpu_ids if gpu_id < all_gpus_length]

    try:
        tf.config.experimental.set_visible_devices(selected_gpus, 'GPU')
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)


def get_gpus_count():
    """
    Return length of available gpus.
    """
    return len(tf.config.experimental.list_logical_devices('GPU'))
