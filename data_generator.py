"""
Data generator class.
"""
import tensorflow as tf
import numpy as np
import os
from omegaconf import DictConfig

from utils.general_utils import join_paths
from utils.images_utils import prepare_image, prepare_mask, image_to_mask_name


class DataGenerator(tf.keras.utils.Sequence):
    """
    Generate batches of data for model by reading images and their
    corresponding masks.
    There are two options you can either pass directory path or list.
    In case of directory, it should contain relative path to images/mask
    folder from project root path.
    In case of list of images, every element should contain absolute path
    for each image and mask.
    Because this generator is also used for prediction, so during testing you can
    set mask path to None if mask are not available for visualization.
    """

    def __init__(self, cfg: DictConfig, mode: str):
        """
        Initialization
        """
        self.cfg = cfg
        self.mode = mode
        self.batch_size = self.cfg.HYPER_PARAMETERS.BATCH_SIZE
        # set seed for reproducibility
        np.random.seed(cfg.SEED)

        # read images from directory
        if isinstance(self.cfg.DATASET[mode].IMAGES_PATH, str):
            # has only images name not full path
            self.images_paths = os.listdir(
                join_paths(
                    self.cfg.WORK_DIR,
                    self.cfg.DATASET[mode].IMAGES_PATH
                )
            )
        else:
            # read images from paths given in list
            # absolute paths of images
            self.images_paths = self.cfg.DATASET[mode].IMAGES_PATH

        self.mask_available = True
        # check mask are available or not
        if self.cfg.DATASET.VAL.MASK_PATH is None:
            self.mask_available = False

        # self.images_paths.sort()  # no need for sorting

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
        return self.__data_generation(indexes)

    def __data_generation(self, indexes):
        """
        Generates batch data
        """

        # create empty array to store batch data
        batch_images = np.zeros(
            (
                self.cfg.HYPER_PARAMETERS.BATCH_SIZE,
                self.cfg.INPUT.HEIGHT,
                self.cfg.INPUT.WIDTH,
                self.cfg.INPUT.CHANNELS
            )
        ).astype(np.float32)

        if self.mask_available:
            batch_masks = np.zeros(
                (
                    self.cfg.HYPER_PARAMETERS.BATCH_SIZE,
                    self.cfg.INPUT.HEIGHT,
                    self.cfg.INPUT.WIDTH,
                    self.cfg.OUTPUT.CLASSES
                )
            ).astype(np.float32)

        for i, index in enumerate(indexes):
            if isinstance(self.cfg.DATASET[self.mode].IMAGES_PATH, str):
                # create full path from folder
                img_path = join_paths(
                    self.cfg.WORK_DIR,
                    self.cfg.DATASET[self.mode].IMAGES_PATH,
                    self.images_paths[index]
                )
                if self.mask_available:
                    mask_path = join_paths(
                        self.cfg.WORK_DIR,
                        self.cfg.DATASET[self.mode].MASK_PATH,
                        image_to_mask_name(self.images_paths[index])
                    )
            else:
                # extract path from list
                img_path = self.images_paths[int(index)]
                if self.mask_available:
                    mask_path = self.cfg.DATASET.VAL.MASK_PATH[int(index)]

            # prepare image for model by resizing and preprocessing it
            image = prepare_image(
                img_path,
                self.cfg.PREPROCESS_DATA.RESIZE,
                self.cfg.PREPROCESS_DATA.IMAGE_PREPROCESSING_TYPE,
            )

            if self.mask_available:
                # prepare image for model by resizing and preprocessing it
                mask = prepare_mask(
                    mask_path,
                    self.cfg.PREPROCESS_DATA.RESIZE,
                    self.cfg.PREPROCESS_DATA.NORMALIZE_MASK,
                )

            # numpy to tensorflow conversion
            if self.mask_available:
                image, mask = tf.numpy_function(
                    self.tf_func,
                    [image, mask],
                    [tf.float32, tf.int32]
                )
            else:
                image = tf.numpy_function(
                    self.tf_func,
                    [image, ],
                    [tf.float32, ]
                )

            # set shape attributes which was lost during Tf conversion
            image.set_shape(
                [
                    self.cfg.INPUT.HEIGHT,
                    self.cfg.INPUT.WIDTH,
                    self.cfg.INPUT.CHANNELS
                ]
            )
            batch_images[i] = image

            if self.mask_available:
                # convert mask into one hot vectors
                # height x width --> height x width x output classes
                mask = tf.one_hot(
                    mask,
                    self.cfg.OUTPUT.CLASSES,
                    dtype=tf.int32
                )
                mask.set_shape(
                    [
                        self.cfg.INPUT.HEIGHT,
                        self.cfg.INPUT.WIDTH,
                        self.cfg.OUTPUT.CLASSES
                    ]
                )
                batch_masks[i] = mask

        if self.mask_available:
            return batch_images, batch_masks
        else:
            return batch_images,

    @staticmethod
    def tf_func(*args):
        return args
