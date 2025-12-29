# Image Classification Pipeline

This project provides an end-to-end pipeline for training an image classifier, running inference on images, testing trained models, splitting datasets, and converting trained models into a format suitable for mobile devices (Android) using **TFLite**.

The pipeline supports four main operations:

* **train** – train and validate a model
* **infer** – run inference on a single image
* **test** – evaluate a trained checkpoint on a test dataset
* **split** – split a dataset into train/validation sets

It also includes a utility to **convert** trained PyTorch models to **TFLite**.

---

## Table of Contents

1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Usage](#usage)

   * [Training](#training)
   * [Inference](#inference)
   * [Testing](#testing)
   * [Dataset Splitting](#dataset-splitting)
   * [Model Conversion (PyTorch to TFLite)](#model-conversion-pytorch-to-tflite)

---

## Requirements

* Python 3.10
* CUDA (optional, used automatically if available)

All Python dependencies are listed in `requirements.txt`.

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Setup

### 1. Dataset Structure

For training, the dataset directory **must** contain `train/` and `val/` subdirectories, each organized by class name:

```
dataset/
├── train/
│   ├── reddit/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   ├── twitter/
│   └── other/
└── val/
    ├── reddit/
    ├── twitter/
    └── other/
```

Each subdirectory name is treated as a class label.

For testing, the dataset only needs class directories (no split required):

```
test_dataset/
├── reddit/
├── twitter/
└── other/
```

---

### 2. Supported Models

The pipeline supports the following architectures:

* **ResNet18**
* **MobileNetV3-Large**

Both models:

* Are pre-trained on ImageNet
* Have their final classification layer replaced to match `num_classes`
* Are fully fine-tuned during training

---

## Usage

The pipeline is controlled via `main.py` using subcommands.

```bash
python main.py <mode> [arguments]
```

Supported modes:

* `train`
* `infer`
* `test`
* `split`

---

## Training

Trains a model using a training and validation dataset.

### Arguments

| Argument           | Description                                              |
| ------------------ | -------------------------------------------------------- |
| `-d, --data`       | Path to dataset directory containing `train/` and `val/` |
| `-t, --model-type` | Model architecture: `resnet` or `mobilenet`              |
| `-e, --epochs`     | Number of epochs (default: 100)                          |
| `-b, --batch-size` | Batch size (default: 8)                                  |
| `--lr`             | Learning rate (default: 0.0001)                          |
| `-p, --patience`   | Early stopping patience (default: 10)                    |
| `--lr-patience`    | LR scheduler patience (default: 3)                       |

### Example

```bash
python main.py train \
  --data dataset \
  --model-type resnet \
  --epochs 50 \
  --batch-size 16
```

### Output

* Model checkpoints saved to `checkpoints/`
* Final trained model saved to `models/`
* Loss and accuracy plots saved to `results/`
* Early stopping and LR scheduling applied automatically

---

## Inference

Runs inference on a single image using a trained model checkpoint.

### Arguments

| Argument            | Description                      |
| ------------------- | -------------------------------- |
| `-m, --model`       | Path to trained model checkpoint |
| `-i, --input-image` | Path to input image              |

### Example

```bash
python main.py infer \
  --model checkpoints/resnet_001.pt \
  --input-image sample.jpg
```

### Output

* Prints the predicted class label for the input image

---

## Testing

Evaluates a trained checkpoint on a labeled test dataset.

### Arguments

| Argument     | Description                    |
| ------------ | ------------------------------ |
| `-d, --data` | Path to test dataset           |
| `--ckpt`     | Path to saved model checkpoint |

### Example

```bash
python main.py test \
  --data test_dataset \
  --ckpt checkpoints/resnet_001.pt
```

### Output

* Validation loss
* Validation accuracy

---

## Dataset Splitting

Splits a dataset into training and validation subsets.

### Arguments

| Argument     | Description                                       |
| ------------ | ------------------------------------------------- |
| `-d, --data` | Path to original dataset                          |
| `-s, --size` | Validation set size (number of samples per class) |
| `-n, --name` | Name of the new dataset directory                 |

### Example

```bash
python main.py split \
  --data raw_dataset \
  --size 100 \
  --name dataset
```

### Output

Creates a new dataset directory with `train/` and `val/` splits.

---

## Model Conversion (PyTorch to TFLite)

Converts a trained PyTorch checkpoint into a TFLite model for mobile deployment.

### ResNet Example

```bash
python convert.py \
  --model checkpoints/screenshot_best.pt \
  --output models/screenshot.tflite \
  --num_classes 3 \
  --model_type resnet
```

### MobileNet Example

```bash
python convert.py \
  --model checkpoints/screenshot_mobile_best.pt \
  --output models/screenshot_mobile.tflite \
  --num_classes 3 \
  --model_type mobilenet
```

### Output

* A `.tflite` model saved at the specified output path

---
