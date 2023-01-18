"""
Evaluation script used to calculate accuracy of trained model
"""
import os
import hydra
from omegaconf import DictConfig
import tensorflow as tf
from tensorflow.keras import mixed_precision

from data_generators import data_generator
from utils.general_utils import join_paths, set_gpus
from models.model import prepare_model
from losses.loss import dice_coef
from losses.unet_loss import unet3p_hybrid_loss


def evaluate(cfg: DictConfig):
    """
    Evaluate or calculate accuracy of given model
    """

    if cfg.USE_MULTI_GPUS.VALUE:
        # change number of visible gpus for evaluation
        set_gpus(cfg.USE_MULTI_GPUS.GPU_IDS)
        # update batch size according to available gpus
        data_generator.update_batch_size(cfg)

    if cfg.OPTIMIZATION.AMP:
        print("Enabling Automatic Mixed Precision(AMP) training")
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    if cfg.OPTIMIZATION.XLA:
        print("Enabling Automatic Mixed Precision(XLA) training")
        tf.config.optimizer.set_jit(True)

    # load training settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE
    )
    # create model
    strategy = None
    if cfg.USE_MULTI_GPUS.VALUE:
        # multi gpu training using tensorflow mirrored strategy
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
        print('Number of visible gpu devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            model = prepare_model(cfg)
    else:
        model = prepare_model(cfg)

    model.compile(
        optimizer=optimizer,
        loss=unet3p_hybrid_loss,
        metrics=[dice_coef],
    )

    # weights model path
    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH,
        f"{cfg.MODEL.WEIGHTS_FILE_NAME}.hdf5"
    )

    assert os.path.exists(checkpoint_path), \
        f"Model weight's file does not exist at \n{checkpoint_path}"

    # TODO: verify without augment it produces same results
    # load model weights
    model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
    model.summary()

    # data generators
    val_generator = data_generator.get_data_generator(cfg, "VAL", strategy)
    validation_steps = data_generator.get_iterations(cfg, mode="VAL")

    # evaluation metric
    evaluation_metric = "dice_coef"
    if len(model.outputs) > 1:
        evaluation_metric = f"{model.output_names[0]}_dice_coef"

    result = model.evaluate(
        x=val_generator,
        steps=validation_steps,
        workers=cfg.DATALOADER_WORKERS,
        return_dict=True,
    )

    # return computed loss, validation accuracy and it's metric name
    return result, evaluation_metric


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Read config file and pass to evaluate method
    """
    result, evaluation_metric = evaluate(cfg)
    print(result)
    print(f"Validation dice coefficient: {result[evaluation_metric]}")


if __name__ == "__main__":
    main()
