# UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhamidriasat%2FUNet-3-Plus&count_bg=%2379C83D&title_bg=%23555555&icon=sega.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)      <a href="/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license" /></a>

<!-- https://hits.seeyoufarm.com/ -->

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unet-3-a-full-scale-connected-unet-for/medical-image-segmentation-on-lits2017)](https://paperswithcode.com/sota/medical-image-segmentation-on-lits2017?p=unet-3-a-full-scale-connected-unet-for)

## Table of Contents

- UNet 3+
    - [Table of Contents](#table-of-contents)
    - [Installation](#installation)
    - [Code Structure](#code-structure)
    - [Config](#config)
    - [Data Preparation](#data-preparation)
    - [Models](#models)
    - [Getting Started](#getting-started)
        - [Inference Demo with Pre-trained Models](#inference-demo-with-pre-trained-models)
        - [Training & Evaluation in Command Line](#training--evaluation-in-command-line)
        - [Multiple Runs](#multiple-runs)

## Installation

**Requirements**

* Python >= 3.6
* [TensorFlow](https://www.tensorflow.org/install) >= 2.4
* CUDA 9.2, 10.0, 10.1, 10.2, 11.0

This code base is tested against above-mentioned Python and TensorFlow versions. But it's expected to work for latest
versions too.

* Install other requirements.

```angular2html
pip install -r requirements.txt
```

## Code Structure

- **checkpoint**: Model checkpoint and logs directory
- **configs**: Configuration file
- **data**: Dataset files (see [Data Preparation](#data-preparation)) for more details
- **data_preparation**: For LiTS data preparation and data verification
- **losses**: Implementations of UNet3+ hybrid loss function and dice coefficient
- **models**:- Unet3+ model files
- **utils**: Generic utility functions
- **data_generator.py**: Data generator for training, validation and testing
- **evaluate.py**: Evaluation script to validate accuracy on trained model
- **predict.ipynb**: Prediction file used to visualize model output inside notebook(helpful for remote server
  visualization)
- **predict.py**: Prediction script used to visualize model output
- **train.py**: Training script

## Config

We are using [Hydra](https://hydra.cc/) for passing configurations. Hydra is a framework for elegantly configuring
complex applications.

Most of the configurations attributes in our [config](configs/config.yaml) are self-explanatory. However, for some
attributes additions comments are added.
You can override configurations from command line too but it's advisable to overrride  

## Data Preparation

- This code can be used to reproduce UNet3+ paper results
  on [LiTS - Liver Tumor Segmentation Challenge](https://competitions.codalab.org/competitions/15595).
- You can also use it to train UNet3+ on custom dataset.

For dataset preparation read [here](/data_preparation/README.md).

## Models

This repo contain all three versions of UNet3+

- [UNet3+ Base model](/models/unet3plus.py)
- [UNet3+ with Deep Supervision](/models/unet3plus_deep_supervision.py)
- [UNet3+ with Deep Supervision and Classification Guided Module](/models/unet3plus_deep_supervision_cgm.py)

[Here](/losses/unet_loss.py) you can find UNet3+ hybrid loss.


> This branch is in development mode. So changes are expected.

TODO List

- [x] Complete README.md
- [ ] Add requirements file
- [ ] Add multiprocessing in LiTS data preprocessing
- [ ] Load data through NVIDIA DALI

Licensed under [MIT License](LICENSE)
