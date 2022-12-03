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
    cv2_img = pil_image_to_array(pil_image)
    return cv2.cvtColor(cv2_img, cv2.COLOR_RGB2BGR)


def opencv_to_pillow_image(opencv_image):
    """
    Convert OpenCV image to Pillow image
    """

    cv2_img = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)
