# Image Classification Pipeline

This project provides an end-to-end pipeline for training an image classifier, running inference on images, and converting the trained model into a format suitable for mobile devices (Android) using **TFLite**.

The pipeline supports two main operations: **training** a model and **running inference**. Additionally, it includes a utility to **convert** the trained PyTorch model to **TFLite** for use in mobile devices.

## Table of Contents
1. [Requirements](#requirements)
2. [Setup](#setup)
3. [Usage](#usage)
    - [Training Mode](#training-mode)
    - [Inference Mode](#inference-mode)
    - [Model Conversion (PyTorch to TFLite)](#model-conversion-pytorch-to-tflite)
4. [Model and Dataset Structure](#model-and-dataset-structure)
    - [Model](#model)
    - [Training Pipeline](#training-pipeline)
5. [Contributing](#contributing)
6. [License](#license)

## Requirements

- Python 3.x
- PyTorch
- TFLite Edge (for model conversion)
- Matplotlib
- Other dependencies can be installed via the `requirements.txt` file.

### Install Dependencies:
```bash
pip install -r requirements.txt
```

## Setup

### 1. Dataset
You need a dataset for training your image classifier. The images should be stored in directories corresponding to their class labels. For example:

```
dataset/
    ├── reddit/
    │   ├── image1.jpg
    │   ├── image2.jpg
    ├── twitter/
    │   ├── image1.jpg
    │   ├── image2.jpg
    └── other/
        ├── image1.jpg
        ├── image2.jpg
```

Make sure the dataset directory follows this structure, where each subdirectory corresponds to a specific class.

### 2. Model
This pipeline uses a pre-trained **ResNet18** model from PyTorch’s `torchvision` library. The model is fine-tuned for your specific dataset by replacing the final fully connected layer to match the number of classes in the dataset. During training, the model weights are updated to fit the provided dataset.

### 3. Class Names File
The `class_names.txt` file should contain the class names, one per line. The order of class names is important, as it determines the model's output labels.

Example `class_names.txt`:
```
reddit
twitter
other
```

## Usage

You can run the pipeline in three modes: `train`, `inference`, and `convert`. Use the `--mode` argument to select the mode.

### Training Mode

To train a new model or continue training an existing one, use the following command:

```bash
python main.py --mode train --dataset /path/to/your/dataset --epochs 100 --batch_size 8 --lr 0.001 --checkpoint checkpoint/screenshot_model --class_names /path/to/class_names.txt
```

#### Arguments:
- `--mode train`: Specifies that you want to train the model.
- `--dataset`: Path to the directory containing your dataset.
- `--class_names`: Path to the text file containing class names.
- `--epochs`: Number of epochs to train the model (optional).
- `--batch_size`: The batch size used during training (optional).
- `--lr`: The learning rate for the optimizer (optional).
- `--checkpoint`: Filename (without ext) to save the trained model's checkpoint (optional).
- `--plot_file`: Path to the file where the loss and accuracy plot will be saved (optional).


#### Output:
- The model will be trained and the loss and accuracy plots will be saved in `results/loss_accuracy_plot.png`.
- The trained model will be saved to the checkpoint file (`checkpoint/screenshot_model`).

**Early Stopping & Learning Rate Scheduling**:
- The training uses **early stopping**: if the test loss doesn’t improve for a specified number of epochs (defined by `patience`), training will stop early.
- **Learning rate scheduling**: The learning rate is reduced if the test loss plateaus, helping improve the model's convergence. The `ReduceLROnPlateau` scheduler is used to decrease the learning rate when the validation loss does not improve after `lr_patience` epochs. The learning rate is reduced by a factor (defined by `factor`), with a minimum limit set by `min_lr`.

### Inference Mode

To use a trained model to classify an image, use the following command:

```bash
python main.py --mode inference --model /path/to/model_checkpoint --input /path/to/image.jpg --class_names /path/to/class_names.txt
```

#### Arguments:
- `--mode inference`: Specifies that you want to run inference.
- `--model`: Path to the model checkpoint file (trained model).
- `--input`: Path to the image file you want to classify.
- `--class_names`: Path to the text file containing class names.

#### Output:
- The predicted class for the input image will be printed on the console.

### Model Conversion (PyTorch to TFLite)

To convert the trained PyTorch model to **TFLite** format for deployment on mobile devices (specifically Android), use the `convert.py` script. This script loads the PyTorch model, converts it to the TFLite format, and saves it as a `.tflite` file.

```bash
python convert.py
```

#### Output:
- A TFLite model file will be generated and saved at the specified path (`models/screenshot_v3.tflite`).

### `convert.py` Details:
- **PyTorch to TFLite Conversion**: This script leverages `ai_edge_torch` to convert the PyTorch model to a TFLite model. The model is set to evaluation mode during conversion, and a sample input tensor is used to trace the model before conversion.
- **Mobile Deployment**: The converted `.tflite` model can be used in Android applications via TensorFlow Lite for on-device inference.


## Model and Dataset Structure

### Model

The model used for classification is based on **ResNet18**, a pre-trained model from `torchvision.models`. The model is fine-tuned by replacing the last fully connected layer to match the number of output classes in the dataset. This allows the model to leverage the learned features from the pre-trained weights while adapting to the specific classes of your dataset.



### Training Pipeline

- **Optimizer**: The model is trained using the **Adam** optimizer, which adapts the learning rate for each parameter.
- **Loss Function**: The **CrossEntropyLoss** criterion is used to calculate the loss between predicted class probabilities and the true labels.
- **Learning Rate Scheduler**: The learning rate is dynamically adjusted using `ReduceLROnPlateau`, which reduces the learning rate if the validation loss plateaus, helping the model converge faster.
- **Early Stopping**: If the validation loss doesn't improve for a specified number of epochs (`patience`), training is halted early to prevent overfitting.


### Dataset
The dataset should consist of images organized into class directories. Each subdirectory represents a class, and the images inside each subdirectory belong to that class. The `Dataset` class in `data.py` is responsible for loading and transforming the dataset.

### Class Names
Class names should be provided in a text file (`class_names.txt`) where each line corresponds to a class name. The order of the class names in the file should match the order used during model training.
