import numpy as np
import cv2
from omegaconf import DictConfig
import matplotlib.pyplot as plt


def read_image(img_path, color_mode):
    return cv2.imread(img_path, color_mode)


def resize_image(img, height, width, resize_method=cv2.INTER_CUBIC):
    """
    Resize image
    """
    return cv2.resize(img, dsize=(width, height), interpolation=resize_method)


def prepare_image(path: str, resize: DictConfig, normalize_type: str):
    image = read_image(path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if resize.VALUE:
        # TODO verify image resizing method
        image = resize_image(image, resize.HEIGHT, resize.WIDTH, cv2.INTER_AREA)

    if normalize_type == "normalize":
        image = image / 255.0

    image = image.astype(np.float32)

    return image


def prepare_mask(path: str, resize: dict, normalize_mask: dict):
    mask = read_image(path, cv2.IMREAD_GRAYSCALE)

    if resize.VALUE:
        mask = resize_image(mask, resize.HEIGHT, resize.WIDTH, cv2.INTER_NEAREST)

    if normalize_mask.VALUE:
        mask = mask / normalize_mask.NORMALIZE_VALUE

    mask = mask.astype(np.int32)

    return mask


def image_to_mask_name(image_name: str):
    # image name--> image_28_0.png
    # mask name--> mask_28_0.png
    return image_name.replace('image', 'mask')


def postprocess_mask(mask):
    mask = np.argmax(mask, axis=-1)
    return mask.astype(np.int32)


def denormalize_mask(mask, classes):
    mask = mask * (255 / classes)
    return mask.astype(np.int32)


def display(display_list, show_true_mask=False):
    """
    Show list of images. it could be
    either [image, true_mask, predicted_mask] or [image, predicted_mask].
    Set show_true_mask to True if true mask is available or vice versa
    """
    if show_true_mask:
        title_list = ('Input Image', 'True Mask', 'Predicted Mask')
        plt.figure(figsize=(12, 4))
    else:
        title_list = ('Input Image', 'Predicted Mask')
        plt.figure(figsize=(8, 4))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        if title_list is not None:
            plt.title(title_list[i])
        if len(np.squeeze(display_list[i]).shape) == 2:
            plt.imshow(np.squeeze(display_list[i]), cmap='gray')
            plt.axis('on')
        else:
            plt.imshow(np.squeeze(display_list[i]))
            plt.axis('on')
    plt.show()
