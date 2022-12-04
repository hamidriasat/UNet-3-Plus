import numpy as np
import cv2
from PIL import Image


def read_pil_image(img_path):
    return Image.open(img_path)


def pil_image_to_array(pil_image):
    return np.array(pil_image)


def pil_to_opencv_image(pil_image):
    """
    Convert Pillow image to OpenCV image
    """
    if not isinstance(pil_image, np.ndarray):
        pil_image = pil_image_to_array(pil_image)
    return cv2.cvtColor(pil_image, cv2.COLOR_RGB2BGR)


def opencv_to_pil_image(opencv_image):
    """
    Convert OpenCV image to Pillow image
    """

    cv2_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)


def resize_image(img, height, width, resize_method=cv2.INTER_CUBIC):
    """
    Resize image
    """
    return cv2.resize(img, dsize=(width, height), interpolation=resize_method)


def read_image(path: str, resize: dict, normalize_type: str):
    image = read_pil_image(path)

    if resize.VALUE:
        image = pil_to_opencv_image(image)
        image = resize_image(image, resize.HEIGHT, resize.WIDTH)
        image = opencv_to_pil_image(image)

    image = pil_image_to_array(image)

