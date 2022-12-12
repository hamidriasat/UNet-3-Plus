import hydra
from omegaconf import DictConfig
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import data_generator
from utils.general_utils import join_paths
from utils.images_utils import postprocess_mask, denormalize_mask
from model.unet3plus import unet_3plus, tiny_unet_3plus


def create_model(cfg: DictConfig):
    return tiny_unet_3plus(
        [
            cfg.INPUT.HEIGHT,
            cfg.INPUT.WIDTH,
            cfg.INPUT.CHANNELS,
        ],
        cfg.OUTPUT.CLASSES
    )


title = ['Input Image', 'True Mask', 'Predicted Mask']


def display(display_list, titlelist=title):
    plt.figure(figsize=(12, 4))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(titlelist[i])
        if len(np.squeeze(display_list[i]).shape) == 2:
            plt.imshow(np.squeeze(display_list[i]), cmap='gray')
            plt.axis('on')
        else:
            plt.imshow(np.squeeze(display_list[i]))
            plt.axis('on')
    plt.show()


def predict(cfg: DictConfig):
    val_generator = data_generator.DataGenerator(cfg, mode="VAL")

    model = create_model(cfg)

    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.CHECKPOINT_PATH,
        # 'model-epoch_{epoch:03d}-val_dice_coef{val_dice_coef:.3f}.hdf5',
        'model.hdf5'
    )
    model.load_weights(checkpoint_path)
    # model.summary()

    showed_images = 0
    for batch_images, batch_mask in val_generator:
        batch_predictions = model.predict_on_batch(batch_images)
        for image, mask, prediction in zip(
                batch_images, batch_mask, batch_predictions):
            mask = postprocess_mask(mask)
            mask = denormalize_mask(mask, cfg.OUTPUT.CLASSES)

            prediction = postprocess_mask(prediction)
            prediction = denormalize_mask(prediction, cfg.OUTPUT.CLASSES)

            if np.unique(mask).shape[0] == 2:
                display([image[:, :, 1], mask, prediction])

            showed_images += 1
        # if showed_images >= 30: break


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    predict(cfg)


if __name__ == "__main__":
    main()
