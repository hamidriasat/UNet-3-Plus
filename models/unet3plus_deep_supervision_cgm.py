# ------------------------------------------------------------------------------
# Written by Hamid Ali (hamidriasat@gmail.com)
# ------------------------------------------------------------------------------
import tensorflow as tf
import tensorflow.keras as k
from .unet3plus_utils import conv_block, dot_product


def unet3plus_deepsup_cgm(INPUT_SHAPE, OUTPUT_CHANNELS, training=False):
    """ UNet_3Plus with Deep Supervision and Classification Guided Module """
    filters = [64, 128, 256, 512, 1024]

    input_layer = k.layers.Input(shape=INPUT_SHAPE, name="input_layer")  # 320*320*3

    """ Encoder"""
    # block 1
    e1 = conv_block(input_layer, filters[0])  # 320*320*64

    # block 2
    e2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 160*160*64
    e2 = conv_block(e2, filters[1])  # 160*160*128

    # block 3
    e3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 80*80*128
    e3 = conv_block(e3, filters[2])  # 80*80*256

    # block 4
    e4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 40*40*256
    e4 = conv_block(e4, filters[3])  # 40*40*512

    # block 5, bottleneck layer
    e5 = k.layers.MaxPool2D(pool_size=(2, 2))(e4)  # 20*20*512
    e5 = conv_block(e5, filters[4])  # 20*20*1024

    """ Classification Guided Module. Part 1"""
    cls = k.layers.Dropout(rate=0.5)(e5)
    cls = k.layers.Conv2D(2, kernel_size=(1, 1), padding="same", strides=(1, 1))(cls)
    cls = k.layers.GlobalMaxPooling2D()(cls)
    cls = k.activations.sigmoid(cls)
    cls = tf.argmax(cls, axis=-1)
    cls = cls[..., tf.newaxis]
    cls = tf.cast(cls, dtype=tf.float32, )

    """ Decoder """
    cat_channels = filters[0]
    cat_blocks = len(filters)
    upsample_channels = cat_blocks * cat_channels

    """ d4 """
    e1_d4 = k.layers.MaxPool2D(pool_size=(8, 8))(e1)  # 320*320*64  --> 40*40*64
    e1_d4 = conv_block(e1_d4, cat_channels, n=1)  # 320*320*64  --> 40*40*64

    e2_d4 = k.layers.MaxPool2D(pool_size=(4, 4))(e2)  # 160*160*128 --> 40*40*128
    e2_d4 = conv_block(e2_d4, cat_channels, n=1)  # 160*160*128 --> 40*40*64

    e3_d4 = k.layers.MaxPool2D(pool_size=(2, 2))(e3)  # 80*80*256  --> 40*40*256
    e3_d4 = conv_block(e3_d4, cat_channels, n=1)  # 80*80*256  --> 40*40*64

    e4_d4 = conv_block(e4, cat_channels, n=1)  # 40*40*512  --> 40*40*64

    e5_d4 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(e5)  # 80*80*256  --> 40*40*256
    e5_d4 = conv_block(e5_d4, cat_channels, n=1)  # 20*20*1024  --> 20*20*64

    d4 = k.layers.concatenate([e1_d4, e2_d4, e3_d4, e4_d4, e5_d4])
    d4 = conv_block(d4, upsample_channels, n=1)  # 40*40*320  --> 40*40*320

    """ d3 """
    e1_d3 = k.layers.MaxPool2D(pool_size=(4, 4))(e1)  # 320*320*64 --> 80*80*64
    e1_d3 = conv_block(e1_d3, cat_channels, n=1)  # 80*80*64 --> 80*80*64

    e2_d3 = k.layers.MaxPool2D(pool_size=(2, 2))(e2)  # 160*160*256 --> 80*80*256
    e2_d3 = conv_block(e2_d3, cat_channels, n=1)  # 80*80*256 --> 80*80*64

    e3_d3 = conv_block(e3, cat_channels, n=1)  # 80*80*512 --> 80*80*64

    e4_d3 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d4)  # 40*40*320 --> 80*80*320
    e4_d3 = conv_block(e4_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

    e5_d3 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(e5)  # 20*20*320 --> 80*80*320
    e5_d3 = conv_block(e5_d3, cat_channels, n=1)  # 80*80*320 --> 80*80*64

    d3 = k.layers.concatenate([e1_d3, e2_d3, e3_d3, e4_d3, e5_d3])
    d3 = conv_block(d3, upsample_channels, n=1)  # 80*80*320 --> 80*80*320

    """ d2 """
    e1_d2 = k.layers.MaxPool2D(pool_size=(2, 2))(e1)  # 320*320*64 --> 160*160*64
    e1_d2 = conv_block(e1_d2, cat_channels, n=1)  # 160*160*64 --> 160*160*64

    e2_d2 = conv_block(e2, cat_channels, n=1)  # 160*160*256 --> 160*160*64

    d3_d2 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d3)  # 80*80*320 --> 160*160*320
    d3_d2 = conv_block(d3_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d4_d2 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d4)  # 40*40*320 --> 160*160*320
    d4_d2 = conv_block(d4_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    e5_d2 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(e5)  # 20*20*320 --> 160*160*320
    e5_d2 = conv_block(e5_d2, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d2 = k.layers.concatenate([e1_d2, e2_d2, d3_d2, d4_d2, e5_d2])
    d2 = conv_block(d2, upsample_channels, n=1)  # 160*160*320 --> 160*160*320

    """ d1 """
    e1_d1 = conv_block(e1, cat_channels, n=1)  # 320*320*64 --> 320*320*64

    d2_d1 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)  # 160*160*320 --> 320*320*320
    d2_d1 = conv_block(d2_d1, cat_channels, n=1)  # 160*160*320 --> 160*160*64

    d3_d1 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)  # 80*80*320 --> 320*320*320
    d3_d1 = conv_block(d3_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    d4_d1 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)  # 40*40*320 --> 320*320*320
    d4_d1 = conv_block(d4_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    e5_d1 = k.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5)  # 20*20*320 --> 320*320*320
    e5_d1 = conv_block(e5_d1, cat_channels, n=1)  # 320*320*320 --> 320*320*64

    d1 = k.layers.concatenate([e1_d1, d2_d1, d3_d1, d4_d1, e5_d1, ])
    d1 = conv_block(d1, upsample_channels, n=1)  # 320*320*320 --> 320*320*320

    """ Deep Supervision Part"""
    # last layer does not have batchnorm and relu
    d1 = conv_block(d1, OUTPUT_CHANNELS, n=1, is_bn=False, is_relu=False)
    if training:
        d2 = conv_block(d2, OUTPUT_CHANNELS, n=1, is_bn=False, is_relu=False)
        d3 = conv_block(d3, OUTPUT_CHANNELS, n=1, is_bn=False, is_relu=False)
        d4 = conv_block(d4, OUTPUT_CHANNELS, n=1, is_bn=False, is_relu=False)
        e5 = conv_block(e5, OUTPUT_CHANNELS, n=1, is_bn=False, is_relu=False)

        # d1 = no need for upsampling
        d2 = k.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(d2)
        d3 = k.layers.UpSampling2D(size=(4, 4), interpolation='bilinear')(d3)
        d4 = k.layers.UpSampling2D(size=(8, 8), interpolation='bilinear')(d4)
        e5 = k.layers.UpSampling2D(size=(16, 16), interpolation='bilinear')(e5)

    """ Classification Guided Module. Part 2"""
    d1 = dot_product(d1, cls)
    d1 = k.activations.sigmoid(d1)

    if training:
        d2 = dot_product(d2, cls)
        d3 = dot_product(d3, cls)
        d4 = dot_product(d4, cls)
        e5 = dot_product(e5, cls)

        d2 = k.activations.sigmoid(d2)
        d3 = k.activations.sigmoid(d3)
        d4 = k.activations.sigmoid(d4)
        e5 = k.activations.sigmoid(e5)

    if training:
        return tf.keras.Model(inputs=input_layer, outputs=[d1, d2, d3, d4, e5, cls], name='UNet3Plus_DeepSup_CGM')
    else:
        return tf.keras.Model(inputs=input_layer, outputs=[d1, ], name='UNet3Plus_DeepSup_CGM')


if __name__ == "__main__":
    """## Model Compilation"""
    INPUT_SHAPE = [320, 320, 1]
    OUTPUT_CHANNELS = 1

    unet_3P = unet3plus_deepsup_cgm(INPUT_SHAPE, OUTPUT_CHANNELS)
    unet_3P.summary()

    # tf.keras.utils.plot_model(unet_3P, show_layer_names=True, show_shapes=True)

    # unet_3P.save("unet_3P.hdf5")
