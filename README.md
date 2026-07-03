# Visual Food Object Detection via Open-Vocabulary Learning

## Introduction

This repository contains the code release for the paper **Visual Food Object Detection via Open-Vocabulary Learning**.

The project is built on top of **Object-Centric OVD** and **Detectron2**, and implements an **open-vocabulary food object detection (OVFD)** framework tailored to fine-grained and cluttered food scenes. The current release focuses on three main components:

- **Dynamic Prompt Distribution Network (DPDN)** for instance-adaptive prompt weighting
- **Density-Aware Multiple Instance Learning (DA-MIL)** for hybrid supervision with image-level food tags
- **strict Food2K semantic-leakage filtering** to reduce overlap between auxiliary weak supervision and benchmark category splits

In addition to the core model, this repository includes dataset registration code, training and evaluation configs, preprocessing scripts, and qualitative visualization assets used in the paper.

## Highlights

- Open-vocabulary food object detection for fine-grained food scenes
- Instance-adaptive prompt weighting via **DPDN**
- Hybrid supervised / weakly supervised learning via **DA-MIL**
- Strict **Food2K semantic-leakage filtering** for auditable weak supervision
- Evaluation on **ZSFooD** with cross-dataset transfer to **UEC-Food256**
- Qualitative comparison figures and paper-related assets

## Method Overview

Our OVFD framework is designed for scalable food localization when category inventories change frequently and exhaustive box annotation is impractical.

The repository implements:

- a **DPDN** module that adaptively reweights attribute-oriented prompt templates according to proposal-level ambiguity
- a **DA-MIL** training strategy that learns from image-level Food2K tags while suppressing background-dominant proposals through objectness reweighting
- a **hybrid supervision pipeline** that combines box-annotated Base classes with leakage-controlled weak supervision
- preprocessing utilities for **Food2K semantic filtering**, prompt construction, and feature preparation

The current codebase is adapted from the Object-Centric OVD framework. Some internal filenames, configs, or intermediate assets may still retain legacy naming conventions from earlier development stages.

## Paper Figures

### Qualitative Comparison

<p align="center">
  <img src="paper/qualitative_comparison_2x5_01.png" alt="qualitative comparison" width="100%">
</p>

## Repository Structure

```text
.
├── configs/                      # training and evaluation configs
├── ovd/                          # core model, dataset, evaluation, ROI head logic
├── preprocessing/                # Food2K filtering, prompt generation, analysis scripts
├── tools/                        # dataset and feature preprocessing scripts
├── utils/                        # auxiliary utilities
├── paper/                        # paper source packages and qualitative figures
├── output/                       # local training outputs and checkpoints
├── detectron2/                   # local Detectron2 source
└── train_net.py                  # main training / evaluation entry
```

## Environment Setup

The upstream codebase was originally developed on top of a Detectron2-style OVD pipeline. A practical environment is:

```bash
conda create -n dpdn python=3.12 -y
conda activate dpdn

pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

cd detectron2
pip install -e .
cd ..
```

## Datasets

Experiments in the paper involve:

- **ZSFooD**: primary benchmark for open-vocabulary food detection
- **UEC-Food256**: cross-dataset evaluation for transfer robustness
- **Food2K**: auxiliary image-level weak-supervision source after strict semantic-leakage filtering

Depending on your local preprocessing pipeline, some paths or cached artifacts may still use earlier folder names. Please update config paths and dataset registration entries accordingly.
Download link for an additional 195 validation data samples:
[additional 195 validation data samples](https://drive.google.com/file/d/1_a2O_w6s7lNU3pd4seOGsRIGeeeltGgf/view?usp=drive_link)
## Dataset Preparation

A typical dataset layout is:

```text
/.../datasets/
├── ZSFooD/
│   ├── train2017/
│   ├── val2017/
│   ├── annotations/
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   │   └── ...
│   └── zero-shot/
│       ├── instances_train2017_seen_*.json
│       ├── instances_val2017_all_*.json
│       └── *_cat_info.json
├── UEC-Food256/
│   ├── images/
│   ├── annotations/
│   └── zero-shot/
└── Food2K/
    ├── images/
    ├── metadata/
    └── filtered_lists/
```

Additional data used in this repository may include:

- food-category description features and prompt resources
- Food2K weak-supervision metadata
- benchmark split files and category information

Before training on another machine, update the absolute paths in the config files and dataset registration code.

## Main Training Config

The main config currently used in this repository is:

```text
configs/coco/COCO_OVD_Base_PIS.yaml
```

Although this filename retains an earlier naming convention, it is the main training config used for the current OVFD code release.

Important path fields that usually need to be edited before running:

- `MODEL.PIS_PROP_PATH`
- `MODEL.VLM_TRAIN_DES_PATH`
- `MODEL.VLM_EVAL_DES_PATH`
- `MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH`
- `MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH`
- dataset root defined in `ovd/datasets/coco_zeroshot.py`

## Training

```bash
python train_net.py \
  --num-gpus 8 \
  --config-file configs/coco/COCO_OVD_Base_PIS.yaml
```

## Evaluation

Example evaluation command:

```bash
python train_net.py \
  --num-gpus 8 \
  --config-file configs/coco/COCO_OVD_Base_PIS.yaml \
  --eval-only \
  MODEL.WEIGHTS /path/to/model_best.pth
```

## Model Weights

The released model weights are available at:

- https://huggingface.co/LTaiQin/DPDN-Det

## Acknowledgement

This codebase is adapted from:

- [Object-Centric OVD](https://github.com/hanoonaR/object-centric-ovd)
- [Detectron2](https://github.com/facebookresearch/detectron2)

We thank the original authors for making their code publicly available.
