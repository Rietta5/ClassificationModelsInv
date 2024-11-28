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

(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()

Xtrain = np.expand_dims(Xtrain,-1)
Xtest = np.expand_dims(Xtest,-1)

Xtrain, Xval, Ytrain, Yval = train_test_split(
    Xtrain, Ytrain, test_size=10000, random_state=666)

for i in [56,128,256]:

    dst_big = trasladar_MNIST(Xtrain, (i,i), 0, 0)
    dst_train = tf.data.Dataset.from_tensor_slices((dst_big, Ytrain)).batch(256//8, drop_remainder=True)

    val = trasladar_MNIST(Xval, (i,i), 0, 0)
    dst_val = tf.data.Dataset.from_tensor_slices((val, Yval)).batch(512//8, drop_remainder=True)


    VGG16 = tf.keras.applications.vgg16.VGG16(
        include_top=False,
        weights='imagenet',
        input_shape=(i,i,3),
    )

    for capa in VGG16.layers:
        capa.trainable = False

    model_VGG16 = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
        tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None)),
        VGG16,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation = "softmax")
    ])

    model_VGG16.compile(optimizer = "adam", metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
    history = model_VGG16.fit(dst_train, epochs = 100, validation_data = dst_val,
                              callbacks = [tf.keras.callbacks.EarlyStopping(patience=5),
                                           tf.keras.callbacks.ModelCheckpoint(filepath=f'VGG16_MNIST_{i}.keras', save_best_only=True)])



