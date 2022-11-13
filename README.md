# UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhamidriasat%2FUNet-3-Plus&count_bg=%2379C83D&title_bg=%23555555&icon=sega.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)      <a href="/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license" /></a>
<!-- https://hits.seeyoufarm.com/ -->

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unet-3-a-full-scale-connected-unet-for/medical-image-segmentation-on-lits2017)](https://paperswithcode.com/sota/medical-image-segmentation-on-lits2017?p=unet-3-a-full-scale-connected-unet-for)

` Hit star ⭐ if you find my work useful. `


## [UNet 3+](https://arxiv.org/abs/2004.08790) for Image Segmentation in Tensorflow Keras.

UNet 3+ is lateset from Unet family, proposed for sementic image segmentation.  it takes advantage of full-scale skip connections and deep supervisions.The full-scale skip connections incorporate low-level details with high-level semantics from feature maps in different scales; while the deep supervision learns hierarchical representations from the full-scale aggregated feature maps.

This repository contain all three versions of **UNet 3+** along with hybrid loss function.

### Code Files
* loss.py &rarr; hybrid loss function for UNet3+
* unet3plus.py &rarr; base model of UNet3+
* unet3plus_deep_supervision.py &rarr; UNet3+ with Deep Supervison
* unet3plus_deep_supervision_cgm.py &rarr; UNet3+ with Deep Supervison and Classification Guided Module
* unet3plus_utils.py &rarr; helper functions

# Architecture
![alt text](https://github.com/hamidriasat/UNet-3-Plus/blob/main/images/unet3p_architecture.png)
![alt text](https://github.com/hamidriasat/UNet-3-Plus/blob/main/images/unet3p_architecture_symbols.png)


# Modules
![alt text](https://github.com/hamidriasat/UNet-3-Plus/blob/main/images/unet3p_modules.png)

# Quantitative Comparison
![alt text](https://github.com/hamidriasat/UNet-3-Plus/blob/main/images/unet3p_results.png)


```
Dependies:
Tensorflow 2.0 or later
```

Licensed under [MIT License](LICENSE)
