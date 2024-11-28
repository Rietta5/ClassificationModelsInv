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

    ins = np.ones((1,i,i,1))
    ModeloRNGAP(ins)
    ModeloRNGAP.compile(metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
    ModeloRNGAP.load_weights(f"ResNet50GAP_MNIST_{i}.keras", skip_mismatch=False)

    # dst_big = trasladar_MNIST(Xtrain, (crop,crop), 0, 0)
    # dst_train = tf.data.Dataset.from_tensor_slices((dst_big, Ytrain)).batch(256//8, drop_remainder=True)
    desps_h = range(-desps,desps+1)
    desps_v = range(-desps,desps+1)

    for desp_h in desps_h:
        for desp_v in desps_v:

            test = trasladar_MNIST(Xtest, (i,i), desp_h = desp_h, desp_v = desp_v)
            dst_test = tf.data.Dataset.from_tensor_slices((test, Ytest)).batch(512//8, drop_remainder=True)

            results = ModeloRNGAP.evaluate(dst_test, return_dict=True)
            metricas[(desp_h, desp_v)] = results
            
        
        

    with open(f"met_ResNet50GAP_{i}.pkl", "wb") as f:
        dump(metricas, f)
    


 

    
