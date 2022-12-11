from datetime import datetime
import hydra
import numpy as np
import tensorflow as tf
import data_generator
from omegaconf import DictConfig, OmegaConf
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger
)

from data_preparation.verify_data import verify_data
from utils.general_utils import create_directory, join_paths
from model.unet3plus import unet_3plus, tiny_unet_3plus
from losses.loss import dice_coef
from losses.unet_loss import unet3p_hybrid_loss


def create_training_folders(cfg: DictConfig):
    create_directory(
        join_paths(
            cfg.WORK_DIR,
            cfg.CALLBACKS.MODEL_CHECKPOINT.CHECKPOINT_PATH
        )
    )
    create_directory(
        join_paths(
            cfg.WORK_DIR,
            cfg.CALLBACKS.TENSORBOARD.TB_LOG_PATH
        )
    )


def create_model(cfg: DictConfig):
    return tiny_unet_3plus(
        [
            cfg.INPUT.HEIGHT,
            cfg.INPUT.WIDTH,
            cfg.INPUT.CHANNELS,
        ],
        cfg.OUTPUT.CLASSES
    )


def train(cfg: DictConfig):
    print("Verifying data ...")
    verify_data(cfg)

    create_training_folders(cfg)

    train_generator = data_generator.DataGenerator(cfg, mode="TRAIN")
    val_generator = data_generator.DataGenerator(cfg, mode="VAL")

    # verify generator
    # for i, (temp_batch_img, temp_batch_mask) in enumerate(val_generator):
    #     print(len(temp_batch_img))
    #     if i >= 3: break

    optimizer = tf.keras.optimizers.Adam(lr=cfg.HYPER_PARAMETERS.LEARNING_RATE)

    # if cfg.USE_MULTI_GPUS.VALUE:
    #     strategy = tf.distribute.MirroredStrategy()
    #     print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    #     with strategy.scope():
    #         model = create_model(cfg)
    #         model.compile(
    #             optimizer=optimizer,
    #             loss=unet3p_hybrid_loss,
    #             metrics=[dice_coef],
    #         )
    # else:
    #     model = create_model(cfg)
    #     model.compile(
    #         optimizer=optimizer,
    #         loss=unet3p_hybrid_loss,
    #         metrics=[dice_coef],
    #     )

    if cfg.USE_MULTI_GPUS.VALUE:
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = create_model(cfg)
    else:
        model = create_model(cfg)

    model.compile(
        optimizer=optimizer,
        loss=unet3p_hybrid_loss,
        metrics=[dice_coef],
    )
    # model.summary()

    # the tensorboard log directory will be a unique subdirectory
    # based on the start time for the run
    tb_log_dir = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.TENSORBOARD.TB_LOG_PATH,
        "{}".format(datetime.now().strftime("%Y.%m.%d.%H.%M.%S"))
    )
    print("TensorBoard directory\n" + tb_log_dir)

    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.CHECKPOINT_PATH,
        'model.hdf5'
    )
    print("Weights Directory\n" + checkpoint_path)

    csv_log_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.CSV_LOGGER.CSV_LOG_PATH,
        'training.csv'
    )
    callbacks = [
        TensorBoard(log_dir=tb_log_dir, write_graph=False, profile_batch=0),
        EarlyStopping(
            patience=cfg.CALLBACKS.EARLY_STOPPING.PATIENCE,
            verbose=cfg.VERBOSE
        ),
        ModelCheckpoint(
            checkpoint_path,
            verbose=cfg.VERBOSE,
            save_weights_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_WEIGHTS_ONLY,
            save_best_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_BEST_ONLY,
            monitor="val_dice_coef",
            mode="max"

        ),
        CSVLogger(
            csv_log_path,
            append=cfg.CALLBACKS.CSV_LOGGER.APPEND_LOGS
        )
    ]

    training_steps = train_generator.__len__()
    validation_steps = val_generator.__len__()

    model.fit(
        x=train_generator,
        steps_per_epoch=training_steps,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=cfg.HYPER_PARAMETERS.EPOCHS,
        batch_size=cfg.HYPER_PARAMETERS.BATCH_SIZE,
        callbacks=callbacks,
        workers=cfg.DATALOADER_WORKERS,
    )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    train(cfg)


if __name__ == "__main__":
    main()
