"""
Evaluation script used to calculate dice coefficient on trained UNet3+ model
"""
import os
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import mixed_precision

from utils.general_utils import join_paths, set_gpus, suppress_warnings
from utils.images_utils import postprocess_mask
from data_generators import data_generator
from models.model import prepare_model
from losses.loss import DiceCoefficient


def evaluate_batch(model, images, masks, classes):
    """
    Make prediction on single batch and return dice coefficient
    """
    # make prediction on batch
    predictions = model.predict_on_batch(images)
    if len(model.outputs) > 1:
        predictions = predictions[0]

    # do postprocessing
    masks = postprocess_mask(masks, classes, float)
    predictions = postprocess_mask(predictions, classes, float)

    # convert to tf tensor
    masks = tf.convert_to_tensor(masks)
    predictions = tf.convert_to_tensor(predictions)

    # because post-processing is done manually, by default classes=2 to fix axis problem
    dice_value = DiceCoefficient(post_processed=False, classes=2)
    dice_value = dice_value(masks, predictions)
    return tf.get_static_value(dice_value)


def evaluate(cfg: DictConfig, mode="VAL"):
    """
    Evaluate or calculate accuracy of given model
    """
    # suppress TensorFlow and DALI warnings
    suppress_warnings()

    # set batch size to one
    cfg.HYPER_PARAMETERS.BATCH_SIZE = 1

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
            model = prepare_model(cfg)
    else:
        model = prepare_model(cfg)

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
    val_generator = data_generator.get_data_generator(cfg, mode, strategy)
    validation_steps = data_generator.get_iterations(cfg, mode)

    progress_bar = tqdm(total=validation_steps)

    results = []
    # for each batch
    for i, (batch_images, batch_mask) in enumerate(val_generator):
        if cfg.USE_MULTI_GPUS.VALUE and cfg.DATA_GENERATOR_TYPE == "DALI_GENERATOR":
            # convert tensorflow.python.distribute.values.PerReplica to tuple
            batch_images = batch_images.values
            batch_mask = batch_mask.values

            for batch_images_, batch_mask_ in zip(batch_images, batch_mask):
                dice_value = evaluate_batch(
                    model,
                    batch_images_,
                    batch_mask_,
                    cfg.OUTPUT.CLASSES
                )
                results.append(dice_value)
        else:
            dice_value = evaluate_batch(
                model,
                batch_images,
                batch_mask,
                cfg.OUTPUT.CLASSES
            )
            results.append(dice_value)

        progress_bar.update(1)
        if i >= validation_steps:
            break

    progress_bar.close()

    return sum(results) / len(results)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """
    Read config file and pass to evaluate method
    """
    result = evaluate(cfg)
    print(f"Validation dice coefficient: {result}")


if __name__ == "__main__":
    main()
