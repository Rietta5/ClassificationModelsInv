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

## Datos

(Xtrain, Ytrain), (Xtest, Ytest) = tf.keras.datasets.mnist.load_data()

Xtrain = np.expand_dims(Xtrain,-1)
Xtest = np.expand_dims(Xtest,-1)

for i, desps in zip([56, 128, 256], [10, 25, 50]):

    metricas = {}


    # VGG16 = tf.keras.applications.vgg16.VGG16(
    #     include_top=False,
    #     weights='imagenet',
    #     input_shape=(crop,crop,3),
    # )

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
    #     tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None)),
    #     VGG16,
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(10, activation = "softmax")
    # ])
    
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
    outputs = layers.Dense(10, activation = "softmax")(GAPFinal)

    ModeloVGGGAP = tf.keras.Model(inputs,outputs)

    prepro = tf.keras.layers.Lambda(lambda x: tf.keras.applications.vgg16.preprocess_input(
            tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None))

    ModeloVGGGAP = tf.keras.Sequential([prepro, ModeloVGGGAP])

    ModeloVGGGAP.compile(optimizer = "adam", metrics=["accuracy"], loss = "sparse_categorical_crossentropy")


    ins = np.ones((1,i,i,1))
    ModeloVGGGAP(ins)
    ModeloVGGGAP.compile(metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
    ModeloVGGGAP.load_weights(f"VGG16GAP_MNIST_{i}.keras", skip_mismatch=False)

    # dst_big = trasladar_MNIST(Xtrain, (crop,crop), 0, 0)
    # dst_train = tf.data.Dataset.from_tensor_slices((dst_big, Ytrain)).batch(256//8, drop_remainder=True)
    desps_h = range(-desps,desps+1)
    desps_v = range(-desps,desps+1)

    for desp_h in desps_h:
        for desp_v in desps_v:

            test = trasladar_MNIST(Xtest, (i,i), desp_h = desp_h, desp_v = desp_v)
            dst_test = tf.data.Dataset.from_tensor_slices((test, Ytest)).batch(512//8, drop_remainder=True)

            results = ModeloVGGGAP.evaluate(dst_test, return_dict=True)
            metricas[(desp_h, desp_v)] = results
        

    with open(f"met_VGG16GAP_{i}.pkl", "wb") as f:
        dump(metricas, f)


 

    
