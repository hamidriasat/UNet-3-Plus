"""
Implementation of different loss functions
"""
import tensorflow as tf
import tensorflow.keras.backend as K


def iou(y_true, y_pred):
    """
    Calculate intersection over union (IoU) between images
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f + y_pred_f - y_true_f * y_pred_f)

    return intersection / union


def iou_loss(y_true, y_pred):
    """
    Jaccard / IoU loss
    """
    return 1 - iou(y_true, y_pred)


def focal_loss(y_true, y_pred):
    """
    Focal loss
    """
    gamma = 2.
    alpha = 4.
    epsilon = 1.e-9

    y_true_c = tf.convert_to_tensor(y_true, tf.float32)
    y_pred_c = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred_c, epsilon)
    ce = tf.multiply(y_true_c, -tf.math.log(model_out))
    weight = tf.multiply(y_true_c, tf.pow(
        tf.subtract(1., model_out), gamma)
                         )
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=-1)
    return tf.reduce_mean(reduced_fl)


def ssim_loss(y_true, y_pred):
    """
    Structural Similarity Index loss
    """
    return 1 - tf.image.ssim(y_true, y_pred, max_val=1)


def dice_coef(y_true, y_pred, smooth=1.e-9):
    """
    Calculate dice coefficient
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)
