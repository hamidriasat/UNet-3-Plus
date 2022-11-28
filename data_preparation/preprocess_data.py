"""
Convert LiTS 2017 (Liver Tumor Segmentation) data into UNet3+ data format
LiTS: https://competitions.codalab.org/competitions/17094
"""
import os
from glob import glob
from pathlib import Path
import numpy as np
import cv2
import nibabel as nib

# sys.path.append(os.path.abspath("../"))  # remove me
BASE_PATH = "H:\\Projects\\UNet3P"  # os.getcwd()


def join_paths(*paths):
    return os.path.normpath(os.path.sep.join(path.rstrip(r"\/") for path in paths))


def create_directory(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_nii(filepath):
    '''
    Reads .nii file and returns pixel array
    '''
    ct_scan = nib.load(filepath).get_fdata()
    ct_scan = np.rot90(np.array(ct_scan))
    return ct_scan


def crop_center(img, croph, cropw):
    """
    Center crop on given height and width
    """
    height, width = img.shape[:2]
    starth = height // 2 - (croph // 2)
    startw = width // 2 - (cropw // 2)
    return img[starth:starth + croph, startw:startw + cropw, :]


def linear_scale(img):
    """
    First convert image to range of 0-1 and them scale to 255
    """
    img = (img - img.min(axis=(0, 1))) / (img.max(axis=(0, 1)) - img.min(axis=(0, 1)))
    return img * 255


def clip_scan(img, min=-200, max=250):
    """
    Clip scan to given range
    """
    return np.clip(img, min, max)


def resize_image(img, height, width, resize_method):
    """
    Resize image
    """
    return cv2.resize(img, dsize=(width, height), interpolation=resize_method)


def resize_scan(scan, new_height, new_width, scan_type):
    """
    Resize CT scan to given size
    :param scan:
    :param new_height:
    :param new_width:
    :return:
    """
    scan_shape = scan.shape
    resized_scan = np.zeros((new_height, new_width, scan_shape[2]), dtype=scan.dtype)
    resize_method = cv2.INTER_CUBIC if scan_type == "image" else cv2.INTER_NEAREST
    for start in range(0, scan_shape[2], scan_shape[1]):
        end = start + scan_shape[1]
        if end >= scan_shape[2]: end = scan_shape[2]
        resized_scan[:, :, start:end] = resize_image(scan[:, :, start:end],
                                                     new_height, new_width,
                                                     resize_method)

    return resized_scan


def save_images(scan, save_path, img_index):
    """
    Based on UNet3+ requirement "input image had three channels, including
    the slice to be segmented and the upper and lower slices, which was
    cropped to 320×320" save each scan as separate image with previous and
    next scan concatenated.
    """
    scan_shape = scan.shape
    for index in range(scan_shape[-1]):
        before_index = index - 1 if (index - 1) > 0 else 0
        after_index = index + 1 if (index + 1) < scan_shape[-1] else scan_shape[-1] - 1

        # swap before_index with after_index, if you want to load this image in
        # correct order using OpenCV, since center index is same, so for now leaving it as it is
        new_img_path = join_paths(save_path, f"image_{img_index}_{index}.png")
        new_image = np.stack((scan[:, :, before_index], scan[:, :, index], scan[:, :, after_index]), axis=-1)

        cv2.imwrite(new_img_path, new_image)  # save the images as .png


def save_mask(scan, save_path, mask_index):
    """
    Save each scan as separate mask
    """
    for index in range(scan.shape[-1]):
        new_mask_path = join_paths(save_path, f"mask_{mask_index}_{index}.png")
        cv2.imwrite(new_mask_path, scan[:, :, index])  # save grey scale image


def extract_images(images_path, save_path, new_height=320, new_width=320, scan_type="image"):
    for image_path in images_path:
        _, index = str(Path(image_path).stem).split("-")

        scan = read_nii(image_path)
        scan = resize_scan(scan, new_height, new_width, scan_type)
        if scan_type == "image":
            scan = clip_scan(scan, )
            scan = linear_scale(scan)
            save_images(scan, save_path, index)
        else:
            # 0 for background/non-lesion, 1 for liver, 2 for lesion/tumor
            # merging label 2 into label 1, because lesion/tumor is part of liver
            scan = np.where(scan != 0, 1, scan)
            # scan = np.where(scan==2, 1, scan)
            scan = np.uint8(scan)
            save_mask(scan, save_path, index)


def extract_paths():
    train_images_names = glob(join_paths(BASE_PATH, "/data/Training Batch 2/volume-*.nii"))
    train_mask_names = glob(join_paths(BASE_PATH, "/data/Training Batch 2/segmentation-*.nii"))

    val_images_names = glob(join_paths(BASE_PATH, "/data/Training Batch 1/volume-*.nii"))
    val_mask_names = glob(join_paths(BASE_PATH, "/data/Training Batch 1/segmentation-*.nii"))

    train_images_names = sorted(train_images_names)
    train_mask_names = sorted(train_mask_names)
    val_images_names = sorted(val_images_names)
    val_mask_names = sorted(val_mask_names)

    train_images_path = join_paths(BASE_PATH, "/data/train/images")
    train_mask_path = join_paths(BASE_PATH, "/data/train/mask")
    val_images_path = join_paths(BASE_PATH, "/data/val/images")
    val_mask_path = join_paths(BASE_PATH, "/data/val/mask")

    create_directory(train_images_path)
    create_directory(train_mask_path)
    create_directory(val_images_path)
    create_directory(val_mask_path)

    extract_images(train_images_names, train_images_path, scan_type="image")
    extract_images(train_mask_names, train_mask_path, scan_type="mask")
    extract_images(val_images_names, val_images_path, scan_type="image")
    extract_images(val_mask_names, val_mask_path, scan_type="mask")


extract_paths()
