from tqdm.auto import tqdm

from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from pickle import dump, load
import matplotlib.pyplot as plt
import cv2
from IPython.display import clear_output
import tensorflow as tf
# import plotly.express as px
from tensorflow.keras import layers
from functools import partial
from PIL import Image
import re

from utils import *
from iqadatasets.datasets import *

import torch
import piq
from sklearn.model_selection import train_test_split

## WandB
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

config = {
     "model": "VGGGAP",
     "batch_size": 256//8,
     "learning_rate": 1e-3,
     "epochs": 1500,
}
wandb.init(project="LPIPS_Mod",
           mode="disabled",
           job_type="training",
           config=config)
config = wandb.config

## Carga datos
dst_train = TID2008("/media/disk/vista/BBDD_video_image/Image_Quality/TID/TID2008/")
dst_train_rdy = dst_train.shuffle(100, reshuffle_each_iteration=True)\
                         .batch(32, num_parallel_calls=tf.data.AUTOTUNE)\
                         .prefetch(1)
img, dist, mos = next(iter(dst_train_rdy))
print(img.shape, dist.shape, mos.shape)

## Semilla aleatoria

tf.keras.utils.set_random_seed(666)

img_shape = (384,512,3)
VGG16 = tf.keras.applications.vgg16.VGG16(
include_top=False,
weights='imagenet',
input_shape=img_shape,
)

for capa in VGG16.layers:
    capa.trainable = False

prepro = tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
        tf.convert_to_tensor(x)*255., data_format=None))
inputs = VGG16.input

GAPMP1 = layers.GlobalAveragePooling2D()(VGG16.layers[3].output)
GAPMP2 = layers.GlobalAveragePooling2D()(VGG16.layers[6].output)
GAPMP3 = layers.GlobalAveragePooling2D()(VGG16.layers[10].output)
GAPMP4 = layers.GlobalAveragePooling2D()(VGG16.layers[14].output)
GAPMP5 = layers.GlobalAveragePooling2D()(VGG16.layers[18].output)

intermediate_gaps = tf.keras.Model(inputs, [GAPMP1, GAPMP2, GAPMP3, GAPMP4, GAPMP5])
intermediate_gaps = tf.keras.Sequential([prepro, intermediate_gaps])

img, dist = tf.keras.Input(img_shape), tf.keras.Input(img_shape)
intermediate_img = layers.Concatenate(axis=-1)(intermediate_gaps(img))
intermediate_dist = layers.Concatenate(axis=-1)(intermediate_gaps(dist))

weights = Weight()
intermediate_img = weights(intermediate_img)
intermediate_dist = weights(intermediate_dist)

outputs = tf.keras.ops.mean((intermediate_img - intermediate_dist)**2, axis=-1)**(1/2)

VGGGAPLPIPS = tf.keras.Model([img, dist],outputs)

VGGGAPLPIPS.compile(optimizer = "adam", metrics=["accuracy"], loss = PearsonCorrelation())
history = VGGGAPLPIPS.fit(dst_train, epochs = 1500, validation_data = dst_train,
                            callbacks = [tf.keras.callbacks.EarlyStopping(patience=25,monitor="val_accuracy"),
                                        tf.keras.callbacks.ModelCheckpoint(filepath=f'VGGGAP_IMA.keras', save_best_only=True,monitor="val_accuracy"),
                                        WandbMetricsLogger(),
                                        WandbModelCheckpoint(filepath="VGGGAP_IMA.keras", save_best_only=True,monitor="val_accuracy")
                                        ])