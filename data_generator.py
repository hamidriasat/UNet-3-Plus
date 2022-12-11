import tensorflow as tf
import numpy as np
import os
from omegaconf import DictConfig

from utils.general_utils import join_paths
from utils.images_utils import prepare_image, prepare_mask


class DataGenerator(tf.keras.utils.Sequence):
    """
    Generate data for model by reading images and their corresponding masks.
    """

    def __init__(self, cfg: DictConfig, mode: str):
        """
        Initialization
        """
        self.cfg = cfg
        self.mode = mode
        self.batch_size = self.cfg.HYPER_PARAMETERS.BATCH_SIZE
        np.random.seed(cfg.SEED)

        self.images_paths = os.listdir(
            join_paths(
                self.cfg.WORK_DIR,
                self.cfg.DATASET[mode].IMAGES_PATH
            )
        )  # has only images name not full path

        self.images_paths.sort()

        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        # Tensorflow problem: on_epoch_end is not being called at the end
        # of each epoch, so forcing on_epoch_end call
        self.on_epoch_end()
        return int(
            np.floor(
                len(self.images_paths) / self.batch_size
            )
        )

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        self.indexes = np.arange(len(self.images_paths))
        if self.cfg.PREPROCESS_DATA.SHUFFLE[self.mode].VALUE:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        # Generate indexes of the batch
        indexes = self.indexes[
                  index * self.batch_size:(index + 1) * self.batch_size
                  ]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def __data_generation(self, indexes):
        """
        Generates batch data
        """

        batch_images = np.zeros(
            (
                self.cfg.HYPER_PARAMETERS.BATCH_SIZE,
                self.cfg.INPUT.HEIGHT,
                self.cfg.INPUT.WIDTH,
                self.cfg.INPUT.CHANNELS
            )
        ).astype(np.float32)
        batch_masks = np.zeros(
            (
                self.cfg.HYPER_PARAMETERS.BATCH_SIZE,
                self.cfg.INPUT.HEIGHT,
                self.cfg.INPUT.WIDTH,
                self.cfg.OUTPUT.CLASSES
            )
        ).astype(np.float32)

        for i, index in enumerate(indexes):
            # Read an image from folder and resize
            img_path = join_paths(
                self.cfg.WORK_DIR,
                self.cfg.DATASET[self.mode].IMAGES_PATH,
                self.images_paths[index]
            )
            # image name--> image_28_0.png
            # mask name--> mask_28_0.png,
            mask_path = join_paths(
                self.cfg.WORK_DIR,
                self.cfg.DATASET[self.mode].MASK_PATH,
                self.images_paths[index].replace('image', 'mask')
            )

            image = prepare_image(
                img_path,
                self.cfg.PREPROCESS_DATA.RESIZE,
                self.cfg.PREPROCESS_DATA.IMAGE_PREPROCESSING_TYPE,
            )
            mask = prepare_mask(
                mask_path,
                self.cfg.PREPROCESS_DATA.RESIZE,
                self.cfg.PREPROCESS_DATA.NORMALIZE_MASK,
            )

            image, mask = tf.numpy_function(
                self.tf_func,
                [image, mask],
                [tf.float32, tf.int32]
            )

            mask = tf.one_hot(
                mask,
                self.cfg.OUTPUT.CLASSES,
                dtype=tf.int32
            )

            image.set_shape(
                [
                    self.cfg.INPUT.HEIGHT,
                    self.cfg.INPUT.WIDTH,
                    self.cfg.INPUT.CHANNELS
                ]
            )
            mask.set_shape(
                [
                    self.cfg.INPUT.HEIGHT,
                    self.cfg.INPUT.WIDTH,
                    self.cfg.OUTPUT.CLASSES
                ]
            )

            batch_images[i] = image
            batch_masks[i] = mask

        return batch_images, batch_masks

    @staticmethod
    def tf_func(x, y):
        return x, y
