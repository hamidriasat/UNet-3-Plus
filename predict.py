import hydra
from omegaconf import DictConfig

import data_generator
from utils.general_utils import join_paths
from utils.images_utils import display
from utils.images_utils import postprocess_mask, denormalize_mask
from models.model import prepare_model


def predict(cfg: DictConfig):
    val_generator = data_generator.DataGenerator(cfg, mode="VAL")

    model = prepare_model(cfg)

    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH,
        f"{cfg.MODEL.WEIGHTS_FILE_NAME}.hdf5"
    )
    model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
    # model.summary()

    showed_images = 0
    # for batch_images, batch_mask in val_generator:
    for batch_data in val_generator:
        batch_images = batch_data[0]
        if cfg.DATASET.VAL.MASK_PATH is not None:
            batch_mask = batch_data[1]

        batch_predictions = model.predict_on_batch(batch_images)
        if len(model.outputs) > 1:
            batch_predictions = batch_predictions[0]

        for index in range(len(batch_images)):

            image = batch_images[index]
            if cfg.SHOW_CENTER_CHANNEL_IMAGE:
                # for UNet3+ show only center channel as image
                image = image[:, :, 1]

            prediction = batch_predictions[index]
            prediction = postprocess_mask(prediction)
            # denormalize mask for better visualization
            prediction = denormalize_mask(prediction, cfg.OUTPUT.CLASSES)

            if cfg.DATASET.VAL.MASK_PATH is not None:
                mask = batch_mask[index]
                mask = postprocess_mask(mask)
                mask = denormalize_mask(mask, cfg.OUTPUT.CLASSES)

            # if np.unique(mask).shape[0] == 2:
            if cfg.DATASET.VAL.MASK_PATH is not None:
                display([image, mask, prediction], show_true_mask=True)
            else:
                display([image, prediction], show_true_mask=False)

            showed_images += 1
        # stop after displaying below number of images
        # if showed_images >= 10: break


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    predict(cfg)


if __name__ == "__main__":
    main()
