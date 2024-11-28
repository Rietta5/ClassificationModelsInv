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

import torch
import piq
from sklearn.model_selection import train_test_split

## Datos

i = 256


def preprocess(path,
               label,
               ):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.image.resize(img, size=(256,256))

        return img, label

df_train = pd.read_csv("imagenet_train.csv")
imgs_train = df_train.path
labels_train = df_train.label_subset

dst_train = tf.data.Dataset.from_tensor_slices((imgs_train, labels_train))\
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(256//8, drop_remainder=True)

df_test = pd.read_csv("imagenet_test.csv")
imgs_test = df_test.path
labels_test = df_test.label_subset

dst_test = tf.data.Dataset.from_tensor_slices((imgs_test, labels_test))\
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(256//8, drop_remainder=True)

df_val = pd.read_csv("imagenet_val.csv")
imgs_val = df_val.path
labels_val = df_val.label_subset

dst_val = tf.data.Dataset.from_tensor_slices((imgs_val, labels_val))\
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(256//8, drop_remainder=True)


VGG16 = tf.keras.applications.vgg16.VGG16(
include_top=False,
weights='imagenet',
input_shape=(i,i,3),
)

for capa in VGG16.layers:
    capa.trainable = False

inputs = tf.keras.Input((i,i,1))
inputs = VGG16.input

GAPMP1 = layers.GlobalAveragePooling2D()(VGG16.layers[3].output)
GAPMP2 = layers.GlobalAveragePooling2D()(VGG16.layers[6].output)
GAPMP3 = layers.GlobalAveragePooling2D()(VGG16.layers[10].output)
GAPMP4 = layers.GlobalAveragePooling2D()(VGG16.layers[14].output)
GAPMP5 = layers.GlobalAveragePooling2D()(VGG16.layers[18].output)

GAPFinal = layers.Concatenate(axis=-1)([GAPMP1,GAPMP2,GAPMP3,GAPMP4,GAPMP5])
outputs = layers.Dense(160, activation = "softmax")(GAPFinal)

ModeloVGGGAP = tf.keras.Model(inputs,outputs)

prepro = tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
        tf.convert_to_tensor(x)*255., data_format=None))

ModeloVGGGAP = tf.keras.Sequential([prepro, ModeloVGGGAP])

ModeloVGGGAP.compile(optimizer = "adam", metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
history = ModeloVGGGAP.fit(dst_train, epochs = 1500, validation_data = dst_val,
                            callbacks = [tf.keras.callbacks.EarlyStopping(patience=5),
                                        tf.keras.callbacks.ModelCheckpoint(filepath=f'VGG16GAP_IMA.keras', save_best_only=True)])


