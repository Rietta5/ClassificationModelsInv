from tqdm.auto import tqdm
import time

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
        .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(256//4, drop_remainder=True)

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

metricas = {}

ResNet50 = tf.keras.applications.resnet50.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(crop,crop,3)
)

model = tf.keras.Sequential([
    tf.keras.layers.Lambda(lambda x: tf.keras.applications.resnet50.preprocess_input(tf.convert_to_tensor(x)*255., data_format=None)),
    ResNet50,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(160, activation = "softmax")
])

ins = np.ones((1,crop,crop,3))
model(ins)
model.compile(metrics=["accuracy"], loss = "sparse_categorical_crossentropy")
model.load_weights(f"ResNet50_IMA.keras", skip_mismatch=False)

#TRASLACION
desps_h = range(-desps,desps+1)
desps_v = range(-desps,desps+1)
total = len(desps_h)*len(desps_v)

i = 1
for desp_h in desps_h:
    for desp_v in desps_v:
        def postprocess(img, label):
            img = crear_mosaico(img)
            img = trasladar2(img, crop=(crop, crop), desp_h=desp_h, desp_v=desp_v)
            return img, label
        dst_test_t = dst_test.map(postprocess)
        # test = trasladar_MNIST(Xtest, (crop,crop), desp_h = desp_h, desp_v = desp_v)
        
        start_calc = time.time()
        results = model.evaluate(dst_test_t, return_dict=True, verbose=0)
        end_calc = time.time()
        
        metricas[(desp_h, desp_v)] = results
        print(f'[{i}/{total} ({desp_h},{desp_v})] Accuracy: {results["accuracy"]} (Time: {end_calc-start_calc:.2f})')
        i = i + 1
        # break
    # break
with open(f"met_ResNet50_IMA.pkl", "wb") as f:
    dump(metricas, f)

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

# escalas  = escalas = [0.1,0.3,0.5,0.6,0.8,1,1.1,1.3,1.5,1.6,1.8,2]
# for escala in escalas:

#     test = escalar(Xtest, escala=escala, size = 56)
#     dst_test = tf.data.Dataset.from_tensor_slices((test, Ytest)).batch(512//8, drop_remainder=True)

#     results = model.evaluate(dst_test, return_dict=True)
#     metricas[escala] = results


# with open(f"met_ResNet50_{crop}_esc.pkl", "wb") as f:
#     dump(metricas, f)






