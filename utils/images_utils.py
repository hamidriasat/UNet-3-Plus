import numpy as np
import cv2


def read_image(img_path, color_mode):
    return cv2.imread(img_path, color_mode)


def resize_image(img, height, width, resize_method=cv2.INTER_CUBIC):
    """
    Resize image
    """
    return cv2.resize(img, dsize=(width, height), interpolation=resize_method)


def prepare_image(path: str, resize: dict, normalize_type: str):
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
