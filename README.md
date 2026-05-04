# ClassificationModelsInv

This repository contains the official implementation of the paper:

**"Parameter-Efficient Architectural Modifications for Translation-Invariant CNNs"**  
Nuria Alabau-Bosque, Jorge Vila-Tomás, Paula Daudén-Oliver, Valero Laparra, and Jesús Malo.  
[[ArXiv](https://arxiv.org/abs/2604.27870)] [[DOI: 10.48550/arXiv.2604.27870](https://doi.org/10.48550/arXiv.2604.27870)]

## Briefing

Standard Convolutional Neural Networks (CNNs) are surprisingly sensitive to small spatial shifts. This work addresses this fragility by strategically inserting **Global Average Pooling (GAP)** layers into standard architectures like VGG-16. This modification decouples feature recognition from spatial location, making the networks translation-invariant by construction while achieving a **98% reduction in trainable parameters** and doubling translational robustness on ImageNet.

## Abstract

Convolutional Neural Networks (CNNs) are widely assumed to be translation-invariant, yet standard architectures exhibit a startling fragility: even a single-pixel shift can drastically degrade performance due to their reliance on spatially dependent fully connected layers. In this work, we resolve this vulnerability by proposing a lightweight 'Online Architecture' strategy. By strategically inserting Global Average Pooling (GAP) layers at various network depths, we effectively decouple feature recognition from spatial location. Using VGG-16 as a primary case study, we demonstrate that this architectural modification achieves a massive 98% reduction in trainable parameters (from 5.2M to just 82K) and a 90% reduction in total network size (138M to 14M). Despite this drastic pruning, our variants maintain competitive Top-1 accuracy on ImageNet (66.4%) while doubling translational robustness, reducing average relative loss from 0.09 to 0.05. Furthermore, our analysis identifies a fundamental limit to invariance: while GAP resolves macroscopic sensitivity, discrete pooling operations introduce a residual periodic aliasing that prevents perfect pixel-level stability. Finally, we extend these findings to Perceptual Image Quality Assessment (IQA) by integrating our invariant backbones into the LPIPS framework. The resulting metric significantly outperforms the retrained baseline in generalization across the KADID-10k dataset (Spearman 0.89 vs. 0.75) and achieves a near-perfect alignment with human psychophysical response curves on the RAID dataset (Spearman 0.95). These results confirm that enforcing architectural invariance is a far more efficient and biologically plausible path to robustness than traditional data augmentation.

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

## Citation

If you find this work useful for your research, please cite our paper:

```bibtex
@misc{alabaubosque2026parameterefficient,
      title={Parameter-Efficient Architectural Modifications for Translation-Invariant CNNs}, 
      author={Nuria Alabau-Bosque and Jorge Vila-Tomás and Paula Daudén-Oliver and Valero Laparra and Jesús Malo},
      year={2026},
      eprint={2604.27870},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2604.27870}, 
}
```
