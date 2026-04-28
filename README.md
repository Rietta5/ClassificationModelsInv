# ClassificationModelsInv

This repository contains a collection of scripts for training and evaluating deep learning classification models (ResNet and VGG) with a focus on their invariance to geometric transformations such as rotation, scaling, and translation. It also includes experiments with Global Average Pooling (GAP) architectures and Learned Perceptual Image Patch Similarity (LPIPS).

## Project Overview

The project explores how standard classification architectures perform when subjected to various image transformations. It compares standard models with modified versions that utilize Global Average Pooling (GAP) to assess improvements in shift and scale invariance.

### Key Features

- **Training Scripts**: Automated training pipelines for ResNet and VGG models on MNIST and ImageNet.
- **Invariance Testing**: Extensive evaluation tools to measure model performance under rotation, scaling, and translation.
- **GAP Architectures**: Implementation of models using Global Average Pooling for enhanced spatial invariance.
- **Perceptual Metrics**: Modified LPIPS metric training using VGG-GAP features.
- **Utility Functions**: A robust `utils.py` containing image manipulation tools (rotation, translation, scaling) and dataset loaders.

## Repository Structure

- `00_Entrenamiento_*.py`: Training scripts for MNIST.
- `01_Entrenamiento_*.py`: Training scripts for ImageNet.
- `02_TestModelos_*.py`: Evaluation scripts for MNIST models under transformations.
- `03_TestModelos_*.py`: Evaluation scripts for ImageNet models under transformations.
- `05_Entrenamiento_LPIPSVGGGAP_TID08.py`: Training a modified LPIPS metric on the TID2008 dataset.
- `utils.py`: Common helper functions for data loading and image processing.

## Installation

Ensure you have Python 3.8+ and the following dependencies installed:

```bash
pip install tensorflow numpy opencv-python pillow tqdm pandas scipy torch piq scikit-learn wandb
```

*Note: Some scripts might require specific dataset paths to be configured.*

## Usage Examples

### Training a Model

To train a ResNet50 model on MNIST with different input sizes:

```bash
python 00_Entrenamiento_ResNet_MNIST.py
```

### Testing Invariance

To evaluate a trained ResNet50 model's performance on MNIST under various scaling factors:

```bash
python 02_TestModelos_ResNet_MNIST.py
```

This script will load the weights, apply transformations (scaling, rotation, translation), and save the results as `.pkl` files.

### Image Manipulation Utilities

You can use the functions in `utils.py` for your own experiments:

```python
from utils import rotar, escalar, trasladar_MNIST
import numpy as np

# Load your data
# data = ...

# Rotate an image by 45 degrees
rotated_img = rotar(data[0], rotacion=45)

# Scale a dataset
scaled_dataset = escalar(data, escala=1.2, size=128)
```

## Datasets

The project supports the following datasets:
- **MNIST**: Loaded via Keras.
- **ImageNet**: Expected in a `./Imagenet` folder.
- **CIFAR-10**: Loaded via Keras.
- **TID2008/TID2013**: Used for perceptual metric experiments.

## License

This project is licensed under the terms provided in the `LICENSE` file.
