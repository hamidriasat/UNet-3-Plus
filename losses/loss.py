"""
Implementation of different loss functions
"""
import tensorflow as tf
import tensorflow.keras.backend as K


def iou(y_true, y_pred, smooth=1):
    """
    Calculate intersection over union (IoU) between images.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
    union = union - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


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
    Structural Similarity Index loss.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1)
    return K.mean(1 - ssim_value, axis=0)


def dice_coef(y_true, y_pred, smooth=1.e-9):
    """
    Calculate dice coefficient.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
