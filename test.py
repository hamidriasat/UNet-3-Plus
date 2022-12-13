import hydra
from omegaconf import DictConfig
import numpy as np

import data_generator
from utils.general_utils import join_paths
from utils.images_utils import display
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
            # denormalize mask for better visualization
            mask = denormalize_mask(mask, cfg.OUTPUT.CLASSES)

            prediction = postprocess_mask(prediction)
            prediction = denormalize_mask(prediction, cfg.OUTPUT.CLASSES)

            if cfg.SHOW_CENTER_CHANNEL_IMAGE:
                # for UNet3+ show only center channel as image
                image = image[:, :, 1]

            # display only those mask which has some nonzero values
            # if np.unique(mask).shape[0] == 2:
            #     display([image, mask, prediction], show_true_mask=True)
            display([image, mask, prediction], show_true_mask=True)

            showed_images += 1
        # stop after displaying below number of images
        # if showed_images >= 30: break


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    predict(cfg)


if __name__ == "__main__":
    main()
