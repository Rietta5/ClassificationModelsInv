from tqdm.auto import tqdm

from pathlib import Path
import numpy as np
import pandas as pd
import scipy
from pickle import dump, load
import cv2
from IPython.display import clear_output
from tensorflow.keras import layers
from functools import partial
from PIL import Image

from utils import *

import torch
import piq


import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

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


ResNet50 = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(i,i,3)
)

for capa in ResNet50.layers:
        capa.trainable = False

model_ResNet50 = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.keras.applications.resnet50.preprocess_input(
                tf.convert_to_tensor(x)*255.)), #, data_format=None
        ResNet50,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(160, activation = "softmax")
])

model_ResNet50.compile(optimizer = "adam", metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
history = model_ResNet50.fit(dst_train, epochs = 1500, validation_data = dst_val,
                            callbacks = [tf.keras.callbacks.EarlyStopping(patience=5),
                                        tf.keras.callbacks.ModelCheckpoint(filepath=f'ResNet_IMA.keras', save_best_only=True)])




