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
crop = 256
desps = 50

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
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(256//4, drop_remainder=True)

df_val = pd.read_csv("imagenet_val.csv")
imgs_val = df_val.path
labels_val = df_val.label_subset

dst_val = tf.data.Dataset.from_tensor_slices((imgs_val, labels_val))\
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(256//4, drop_remainder=True)

# for crop, desps in zip([56, 128, 256], [10, 25, 50]):
crop = 256
i = 256

metricas = {}

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

ins = np.ones((1,i,i,3))
ModeloVGGGAP(ins)
ModeloVGGGAP.compile(metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
ModeloVGGGAP.load_weights(f"VGG16GAP_IMA.keras", skip_mismatch=False)

 
#TRASLACION
desps_h = range(-desps,desps+1)
desps_v = range(-desps,desps+1)

for desp_h in desps_h:
    for desp_v in desps_v:
        def postprocess(img, label):
            img = crear_mosaico(img)
            img = trasladar2(img, crop=(crop, crop), desp_h=desp_h, desp_v=desp_v)
            return img, label
        dst_test_t = dst_test.map(postprocess)

        results = ModeloVGGGAP.evaluate(dst_test_t, return_dict=True)
        metricas[(desp_h, desp_v)] = results

with open(f"met_VGGGAP_IMA.pkl", "wb") as f:
    dump(metricas, f)







    


 

    
