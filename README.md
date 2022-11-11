# UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhamidriasat%2FUNet-3-Plus&count_bg=%2379C83D&title_bg=%23555555&icon=sega.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)      <a href="/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license" /></a>
<!-- https://hits.seeyoufarm.com/ -->

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unet-3-a-full-scale-connected-unet-for/medical-image-segmentation-on-lits2017)](https://paperswithcode.com/sota/medical-image-segmentation-on-lits2017?p=unet-3-a-full-scale-connected-unet-for)

` Hit star ⭐ if you find my work useful. `


## [UNet 3+](https://arxiv.org/abs/2004.08790) for Image Segmentation in Tensorflow Keras.

UNet 3+ is lateset from Unet family, proposed for sementic image segmentation.  it takes advantage of full-scale skip connections and deep supervisions.The full-scale skip connections incorporate low-level details with high-level semantics from feature maps in different scales; while the deep supervision learns hierarchical representations from the full-scale aggregated feature maps.


model file contain three different implementation of UNet 3+. 
- UNet_3Plus              ==> UNet_3Plus
- UNet_3Plus_DeepSup      ==> UNet_3Plus with Deep Supervison
- UNet_3Plus_DeepSup_CGM  ==> UNet_3Plus with Deep Supervison and Classification Guided Module

### Code Files
* unet3plus.py ==> UNet 3+ model code full code


```
Dependies:
Tensorflow 2.0 or later
```

Licensed under [MIT License](LICENSE)
