import hydra
from omegaconf import DictConfig
import tensorflow as tf

import data_generator
from utils.general_utils import join_paths, set_gpus
from model.unet3plus import unet_3plus, tiny_unet_3plus
from losses.loss import dice_coef
from losses.unet_loss import unet3p_hybrid_loss


def create_model(cfg: DictConfig):
    return tiny_unet_3plus(
        [
            cfg.INPUT.HEIGHT,
            cfg.INPUT.WIDTH,
            cfg.INPUT.CHANNELS,
        ],
        cfg.OUTPUT.CLASSES
    )


def evaluate(cfg: DictConfig):
    if cfg.USE_MULTI_GPUS.VALUE:
        set_gpus(cfg.USE_MULTI_GPUS.GPU_IDS)

    val_generator = data_generator.DataGenerator(cfg, mode="VAL")

    optimizer = tf.keras.optimizers.Adam(lr=cfg.HYPER_PARAMETERS.LEARNING_RATE)
    if cfg.USE_MULTI_GPUS.VALUE:
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
        print('Number of visible gpu devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = create_model(cfg)
    else:
        model = create_model(cfg)

    model.compile(
        optimizer=optimizer,
        loss=unet3p_hybrid_loss,
        metrics=[dice_coef],
    )

    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.CHECKPOINT_PATH,
        # 'model-epoch_{epoch:03d}-val_dice_coef{val_dice_coef:.3f}.hdf5',
        'model.hdf5'
    )
    # model = tf.keras.models.load_model(checkpoint_path)
    model.load_weights(checkpoint_path)
    # model.summary()

    result = model.evaluate(
        x=val_generator,
        batch_size=cfg.HYPER_PARAMETERS.BATCH_SIZE,
        workers=cfg.DATALOADER_WORKERS,
        return_dict=True,
    )

    return result


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print(evaluate(cfg))


if __name__ == "__main__":
    main()
