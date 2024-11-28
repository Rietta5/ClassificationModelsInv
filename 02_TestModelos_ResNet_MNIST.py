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

# for crop, desps in zip([56, 128, 256], [10, 25, 50]):

for crop in [56]:

    metricas = {}

    ResNet50 = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(crop,crop,3)
    )

    model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.keras.applications.resnet50.preprocess_input(
        tf.image.grayscale_to_rgb(tf.convert_to_tensor(x)), data_format=None)),
        ResNet50,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10, activation = "softmax")
    ])

    ins = np.ones((1,crop,crop,1))
    model(ins)
    model.compile(metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
    model.load_weights(f"ResNet50_MNIST_{crop}.keras", skip_mismatch=False)

    #TRASLACION
    # desps_h = range(-desps,desps+1)
    # desps_v = range(-desps,desps+1)

    # for desp_h in desps_h:
    #     for desp_v in desps_v:

    #         test = trasladar_MNIST(Xtest, (crop,crop), desp_h = desp_h, desp_v = desp_v)
    #         dst_test = tf.data.Dataset.from_tensor_slices((test, Ytest)).batch(512//8, drop_remainder=True)

    #         results = model.evaluate(dst_test, return_dict=True)
    #         metricas[(desp_h, desp_v)] = results
    
    # with open(f"met_ResNet50_{crop}.pkl", "wb") as f:
    #     dump(metricas, f)

    #ROTACION
            
    # rotaciones = np.arange(0,11, 0.5)

    # for rot in rotaciones:

    #     test = rotar(Xtest, rotacion = rot)
    #     dst_test = tf.data.Dataset.from_tensor_slices((test, Ytest)).batch(512//8, drop_remainder=True)

    #     results = model.evaluate(dst_test, return_dict=True)
    #     metricas[rot] = results
    
    # with open(f"met_ResNet50_{crop}_rot.pkl", "wb") as f:
    #     dump(metricas, f)
            
    #ESCALA
    
    escalas  = escalas = [0.1,0.3,0.5,0.6,0.8,1,1.1,1.3,1.5,1.6,1.8,2]
    for escala in escalas:

        test = escalar(Xtest, escala=escala, size = 56)
        dst_test = tf.data.Dataset.from_tensor_slices((test, Ytest)).batch(512//8, drop_remainder=True)

        results = model.evaluate(dst_test, return_dict=True)
        metricas[escala] = results
    

    with open(f"met_ResNet50_{crop}_esc.pkl", "wb") as f:
        dump(metricas, f)
    


 

    
