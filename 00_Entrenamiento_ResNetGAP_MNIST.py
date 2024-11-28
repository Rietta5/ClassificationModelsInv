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

    ResNet50 = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(i,i,3)
    )

    for capa in ResNet50.layers:
        capa.trainable = False

    model_ResNet50 = tf.keras.Sequential([
        ResNet50,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation = "softmax")
    ])

    capas_out = []

    for j,layer in enumerate(ResNet50.layers):
        if layer.name.endswith("out"):
            print(j,layer.name)
            capas_out.append(layer)

    inputs = ResNet50.input
    GAPS = [layers.GlobalAveragePooling2D()(l.output) for l in capas_out]

    GAPFinal = layers.Concatenate(axis=-1)(GAPS)

    outputs = layers.Dense(10, activation = "softmax")(GAPFinal)

    ModeloRNGAP = tf.keras.Model(inputs,outputs)
    prepro = tf.keras.layers.Lambda(lambda x: tf.keras.applications.resnet50.preprocess_input(
            tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None))

    ModeloRNGAP = tf.keras.Sequential([prepro, ModeloRNGAP])


    ModeloRNGAP.compile(optimizer = "adam", metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
    history = ModeloRNGAP.fit(dst_train, epochs = 100, validation_data = dst_val,
                              callbacks = [tf.keras.callbacks.EarlyStopping(patience=5),
                                           tf.keras.callbacks.ModelCheckpoint(filepath=f'ResNet50GAP_MNIST_{i}.keras', save_best_only=True)])


