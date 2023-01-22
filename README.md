# UNet 3+: A Full-Scale Connected UNet for Medical Image Segmentation [![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fhamidriasat%2FUNet-3-Plus&count_bg=%2379C83D&title_bg=%23555555&icon=sega.svg&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)      <a href="/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="license" /></a>

<!-- https://hits.seeyoufarm.com/ -->

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/unet-3-a-full-scale-connected-unet-for/medical-image-segmentation-on-lits2017)](https://paperswithcode.com/sota/medical-image-segmentation-on-lits2017?p=unet-3-a-full-scale-connected-unet-for)

` Hit star ⭐ if you find my work useful. `

## Table of Contents

- [UNet 3+](https://arxiv.org/abs/2004.08790) for Image Segmentation in Tensorflow Keras.
    - [Table of Contents](#table-of-contents)
    - [Feature Support Matrix](#feature-support-matrix)
    - [Installation](#installation)
    - [Code Structure](#code-structure)
    - [Config](#config)
    - [Data Preparation](#data-preparation)
    - [Models](#models)
    - [Training & Evaluation](#training--evaluation)
    - [Inference Demo](#inference-demo)

## Feature Support Matrix

The following features are supported by our code base:

|                  Feature                   | UNet3+ Supports |
|:------------------------------------------:|:---------------:|
|                    DALI                    |     &check;     |
|       TensorFlow Multi-GPU Training        |     &check;     |
| TensorFlow Automatic Mixed Precision(AMP)  |     &check;     |
| TensorFlow Accelerated Linear Algebra(XLA) |     &check;     |

#### [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/index.html)

The NVIDIA Data Loading Library (DALI) is a library for data loading and
pre-processing to accelerate deep learning applications. It provides a
collection of highly optimized building blocks for loading and processing
image, video and audio data. It can be used as a portable drop-in
replacement for built in data loaders and data iterators in popular deep
learning frameworks.

#### [TensorFlow Multi-GPU Training](https://www.tensorflow.org/guide/distributed_training)

Distribute training across multiple GPUs, multiple machines, or TPUs.

#### [TensorFlow Automatic Mixed Precision(AMP)](https://www.tensorflow.org/guide/mixed_precision)

Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training to make it run
faster and use less memory. By keeping certain parts of the model in the 32-bit types for numeric stability, the model
will have a lower step time and train equally as well in terms of the evaluation metrics such as accuracy.

#### [TensorFlow Accelerated Linear Algebra(XLA)](https://www.tensorflow.org/xla)

In a TensorFlow program, all of the operations are executed individually by the TensorFlow executor. Each TensorFlow
operation has a precompiled GPU kernel implementation that the executor dispatches to.
XLA provides an alternative mode of running models: it compiles the TensorFlow graph into a sequence of computation
kernels generated specifically for the given model. Because these kernels are unique to the model, they can exploit
model-specific information for optimization.

## Installation

**Requirements**

* Python >= 3.6
* [TensorFlow](https://www.tensorflow.org/install) >= 2.4
* CUDA 9.2, 10.0, 10.1, 10.2, 11.0

This code base is tested against above-mentioned Python and TensorFlow versions. But it's expected to work for latest
versions too.

* Clone code

```
git clone https://github.com/hamidriasat/UNet-3-Plus.git UNet3P
cd UNet3P
```

* Install other requirements.

```
pip install -r requirements.txt
```

## Code Structure

- **checkpoint**: Model checkpoint and logs directory
- **configs**: Configuration file
- **data**: Dataset files (see [Data Preparation](#data-preparation)) for more details
- **data_generators**: Data loaders for UNet3+
- **data_preparation**: For LiTS data preparation and data verification
- **losses**: Implementations of UNet3+ hybrid loss function and dice coefficient
- **models**: Unet3+ model files
- **utils**: Generic utility functions
- **data_generator.py**: Data generator for training, validation and testing
- **evaluate.py**: Evaluation script to validate accuracy on trained model
- **predict.ipynb**: Prediction file used to visualize model output inside notebook(helpful for remote server
  visualization)
- **predict.py**: Prediction script used to visualize model output
- **train.py**: Training script

## Config

Configurations are passed through `yaml` file. For more details on config file read [here](/configs/).

## Data Generators

We support two types of data loaders. `NVIDIA DALI` and `TensorFlow Sequence`
generators. For more details on supported generator types read [here](/data_generators/).

## Data Preparation

- This code can be used to reproduce UNet3+ paper results
  on [LiTS - Liver Tumor Segmentation Challenge](https://competitions.codalab.org/competitions/15595).
- You can also use it to train UNet3+ on custom dataset.

For dataset preparation read [here](/data_preparation/README.md).

## Models

UNet 3+ is lateset from Unet family, proposed for sementic image segmentation. it takes advantage of full-scale skip
connections and deep supervisions.The full-scale skip connections incorporate low-level details with high-level
semantics from feature maps in different scales; while the deep supervision learns hierarchical representations from the
full-scale aggregated feature maps.
![alt text](/figures/unet3p_architecture.png)
Figure 1. UNet3+ architecture diagram from original paper

This repo contains all three versions of UNet3+.

[//]: # (https://stackoverflow.com/questions/47344571/how-to-draw-checkbox-or-tick-mark-in-github-markdown-table)

| #   |                          Description                          |                             Model Name                             | Training Supported |
|:----|:-------------------------------------------------------------:|:------------------------------------------------------------------:|:------------------:|
| 1   |                       UNet3+ Base model                       |                 [unet3plus](/models/unet3plus.py)                  |      &check;       |
| 2   |                 UNet3+ with Deep Supervision                  |     [unet3plus_deepsup](/models/unet3plus_deep_supervision.py)     |      &check;       |
| 3   | UNet3+ with Deep Supervision and Classification Guided Module | [unet3plus_deepsup_cgm](/models/unet3plus_deep_supervision_cgm.py) |      &cross;       |

[Here](/losses/unet_loss.py) you can find UNet3+ hybrid loss.

### Training & Evaluation

To train a model call `train.py` with required model type and configurations .

e.g. To train on base model run

```
python train.py MODEL.TYPE=unet3plus
```

To evaluate the trained models call `evaluate.py`.

e.g. To calculate accuracy of trained UNet3+ Base model on validation data run

```
python evaluate.py MODEL.TYPE=unet3plus
```

**Multi Gpu Training**

Our code support multi gpu training
using [Tensorflow Distributed MirroredStrategy](https://www.tensorflow.org/api_docs/python/tf/distribute/MirroredStrategy)
.
By default, training and evaluation is done on only one gpu. To enable multiple gpus you have to explicitly set
`USE_MULTI_GPUS` values.
e.g. To train on all available gpus run

```
python train.py ... USE_MULTI_GPUS.VALUE=True USE_MULTI_GPUS.GPU_IDS=-1 
```

For `GPU_IDS` two options are available. It could be either integer or list of integers.

- In case Integer:
    - If integer value is -1 then it uses all available gpus.
    - Otherwise, if positive number, then use given number of gpus.
- In case list of Integers: each integer will be considered as gpu id
  e.g. [4,5,7] means use gpu 5,6 and 8 for training/evaluation

### Inference Demo

For visualization two options are available

1. [Visualize from directory](#visualize-from-directory)
2. [Visualize from list](#visualize-from-list)

In both cases mask is optional

You can visualize results through [predict.ipynb](/predict.ipynb) notebook, or you can also override these settings
through command line and call `predict.py`

1. ***Visualize from directory***

In case of visualization from directory, it's going to make prediction and show all images from given directory.
Override the validation data paths and make sure the directory paths are relative to the project base/root path e.g.

```
python predict.py MODEL.TYPE=unet3plus ^
DATASET.VAL.IMAGES_PATH=/data/val/images/ ^
DATASET.VAL.MASK_PATH=/data/val/mask/
```

2. ***Visualize from list***

In case of visualization from list, each list element should contain absolute path of image/mask. For absolute paths
training data naming convention does not matter you can pass whatever naming convention you have, just make sure images,
and it's corresponding mask are on same index.

e.g. To visualize model results on two images along with their corresponding mask, run

```
python predict.py MODEL.TYPE=unet3plus ^
DATASET.VAL.IMAGES_PATH=[^
H:\\Projects\\UNet3P\\data\\val\images\\image_0_48.png,^
H:\\Projects\\UNet3P\\data\\val\images\\image_0_21.png^
] DATASET.VAL.MASK_PATH=[^
H:\\Projects\\UNet3P\\data\\val\\mask\\mask_0_48.png,^
H:\\Projects\\UNet3P\\data\\val\\mask\\mask_0_21.png^
]
```

For custom data visualization set `SHOW_CENTER_CHANNEL_IMAGE=False`. This should set True for only UNet3+ LiTS data.

These commands are tested on Windows. For Linux replace `^` with \ and replace `H:\\Projects` with your own base path

> Note: Don't add space between list elements, it will create problem with Hydra.



In both cases if mask is not available just set the mask path to None

```
python predict.py DATASET.VAL.IMAGES_PATH=... DATASET.VAL.MASK_PATH=None
```

> This branch is in development mode. So changes are expected.

TODO List

- [x] Complete README.md
- [x] Add requirements file
- [ ] Add Data augmentation
- [x] Add multiprocessing in LiTS data preprocessing
- [x] Load data through NVIDIA DALI

We appreciate any feedback so reporting problems, and asking questions are welcomed here.

Licensed under [MIT License](LICENSE)
