# Winning Solution Malawi (Sheoran part)

## Install

Prerequists:

Anaconda

pip

Nvidia 555.42.02 driver

Ubuntu 22.04

Cuda 11.7 in /usr/local (optional, try this if cuda related error occurs)

```
conda create -n arm_mmdet anaconda python=3.9
```
You can activate this newly created env with ```conda activate arm_mmdet```

This enviornment comes with a lot of common packages already installed

For more needed packages, run install.sh file which installs more packages needed for mmdetection
```
./install.sh
```
If you see permission denied error for any .sh file, run ```chmod +x install.sh``` to give it permissions

```
cd Co-DETR/

pip install -v -e . --no-cache-dir
```

Now, come back a little to the folder we were previously in
```
cd ..
```

1. Make a master train file that have information in a more organized way and assign each image to a fold, also splits the data in data/train_images and data/test_images from data/Images folder for better understanding 

```
python initial_data_prep.py
```

2. Create data for MMDET in json file, I only use Fold 0 as it takes long time to train

```
python create_mmdet_data.py
```

## Inference

Start with downloading model weights

```
kaggle datasets download -d harshitsheoran/malawi-final-weights-sheoran
```
```
unzip malawi-final-weights-sheoran.zip -d final_weights/
```

Before we run inference, we confirm that ```./data/test_images``` folder exists, this is where the inference will happen

Now, to actually run inference
```
python inference_mmdet.py
```
This will create submission.csv when completed

## Reproduce Training

Note that to reproduce the training, it is recommended to use 4x 3090 gpus

0. Make sure that Steps to install are followed

We start training model using mmdetection

```
cd Co-DETR/

```

We need to download pretrained checkpoints trained on coco, available publically in https://github.com/Sense-X/Co-DETR/tree/main 

```
./download_pretrain.sh
```

Because exp_002 model is trained with 3 classes, it uses mmdet_data_v2 (created from create_mmdet_data.py), to mod mmdet accordingly

```
./mod_mmdet_for_3_classes.sh
```

```
bash tools/dist_train.sh ./work_dirs/exp_002/co_deformable_detr_swin_base_3x_coco.py 4 ./work_dirs/exp_002_re/ --deterministic --seed 0
```

Because exp_004 is trained with only Tin and Thatch class, let's mod mmdet to support 2 classes

```
./mod_mmdet_for_2_classes.sh
```

```
bash tools/dist_train.sh ./work_dirs/exp_004/co_dino_5scale_swin_large_16e_o365tococo.py 4 ./work_dirs/exp_004_re/ --deterministic --seed 0
```