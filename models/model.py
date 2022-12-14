from omegaconf import DictConfig

from .unet3plus import unet3plus, tiny_unet3plus
from .unet3plus_deep_supervision import unet3plus_deepsup
from .unet3plus_deep_supervision_cgm import unet3plus_deepsup_cgm


def prepare_model(cfg: DictConfig):
    
    if cfg.MODEL.TYPE == "tiny_unet3plus":
        return tiny_unet3plus(
            [
                cfg.INPUT.HEIGHT,
                cfg.INPUT.WIDTH,
                cfg.INPUT.CHANNELS,
            ],
            cfg.OUTPUT.CLASSES
        )
    elif cfg.MODEL.TYPE == "unet3plus":
        return unet3plus(
            [
                cfg.INPUT.HEIGHT,
                cfg.INPUT.WIDTH,
                cfg.INPUT.CHANNELS,
            ],
            cfg.OUTPUT.CLASSES
        )
    elif cfg.MODEL.TYPE == "unet3plus_deepsup":
        return unet3plus_deepsup(
            [
                cfg.INPUT.HEIGHT,
                cfg.INPUT.WIDTH,
                cfg.INPUT.CHANNELS,
            ],
            cfg.OUTPUT.CLASSES
        )
    elif cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        if cfg.OUTPUT.CLASSES != 1:
            raise ValueError(
            "UNet3+ with Deep Supervison and Classification Guided Module"
            "\nOnly works when model output classes are equal to 1"
            )
        return unet3plus_deepsup_cgm(
            [
                cfg.INPUT.HEIGHT,
                cfg.INPUT.WIDTH,
                cfg.INPUT.CHANNELS,
            ],
            cfg.OUTPUT.CLASSES
        )
    else:
        raise ValueError(
            "Wrong model type passed."
            "\nPlease check config file for possible options."
        )
    

