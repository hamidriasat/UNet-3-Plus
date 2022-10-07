# UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation


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
