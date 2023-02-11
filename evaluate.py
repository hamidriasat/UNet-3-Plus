"""
Evaluation script used to calculate accuracy of trained model
"""
import os
import numpy as np
import hydra
from omegaconf import DictConfig
import tensorflow as tf
from tensorflow.keras import mixed_precision

from data_generators import data_generator
from utils.general_utils import join_paths, set_gpus
from models.model import prepare_model
from losses.loss import DiceCoefficient
from losses.unet_loss import unet3p_hybrid_loss
from callbacks.timing_callback import TimingCallback


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

    # create model
    strategy = None
    if cfg.USE_MULTI_GPUS.VALUE:
        # multi gpu training using tensorflow mirrored strategy
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
        print('Number of visible gpu devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE
            )  # optimizer
            if cfg.OPTIMIZATION.AMP:
                optimizer = mixed_precision.LossScaleOptimizer(
                    optimizer,
                    dynamic=True
                )
            model = prepare_model(cfg, training=True)
    else:
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE
        )  # optimizer
        if cfg.OPTIMIZATION.AMP:
            optimizer = mixed_precision.LossScaleOptimizer(
                optimizer,
                dynamic=True
            )
        model = prepare_model(cfg, training=True)

    model.compile(
        optimizer=optimizer,
        loss=unet3p_hybrid_loss,
        metrics=[
            DiceCoefficient(post_processed=True, classes=cfg.OUTPUT.CLASSES)
        ],
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
    timing_callback = TimingCallback()

    result = model.evaluate(
        x=val_generator,
        steps=validation_steps,
        callbacks=[timing_callback],
        workers=cfg.DATALOADER_WORKERS,
        return_dict=True,
    )

    # return computed loss, validation accuracy, metric name, prediction time
    return result, evaluation_metric, timing_callback.prediction_time


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Read config file and pass to evaluate method
    """
    result, evaluation_metric, time_taken = evaluate(cfg)
    print(result)
    print(f"Validation dice coefficient: {result[evaluation_metric]}")

    mean_time = np.mean(time_taken)
    mean_fps = 1 / mean_time
    print(f"Mean Time: {mean_time:1.7f} - Mean FPS: {mean_fps:1.7f}")

    avg_step_time = np.mean(time_taken)
    print("\nAverage step time: %.1f msec" % (avg_step_time * 1e3))
    print("Average throughput: %d samples/sec" % (1 / avg_step_time))


if __name__ == "__main__":
    main()
