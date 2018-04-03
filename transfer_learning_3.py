'''
Created on Mar 27, 2018

@author: kaliaanup
'''
from keras.layers import Input, Dense
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D


img_width, img_height = 256, 256

### Build the network 
img_input = Input(shape=(256, 256, 3))
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

model = Model(input = img_input, output = x)

model.summary()

layer_dict = dict([(layer.name, layer) for layer in model.layers])
[layer.name for layer in model.layers]

import h5py
weights_path = 'trained_model/vgg19_weights.h5' 
f = h5py.File(weights_path)
list(f["model_weights"].keys())


# list all the layer names which are in the model.
layer_names = [layer.name for layer in model.layers]

# Here we are extracting model_weights for each and every layer from the .h5 file
f["model_weights"]["block1_conv1"].attrs["weight_names"]
# we are assiging this array to weight_names below 
f["model_weights"]["block1_conv1"]["block1_conv1/kernel:0"]
# The list comprehension (weights) stores these two weights and bias of both the layers 
layer_names.index("block1_conv1")
model.layers[1].set_weights(weights)
#With a for loop we can set_weights for the entire network.

for i in layer_dict.keys():
    weight_names = f["model_weights"][i].attrs["weight_names"]
    weights = [f["model_weights"][i][j] for j in weight_names]
    index = layer_names.index(i)
    model.layers[index].set_weights(weights)

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import glob

features = []
for i in tqdm(files_location):
        im = cv2.imread(i)
        im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (256, 256)).astype(np.float32) / 255.0
        im = np.expand_dims(im, axis =0)
        outcome = model_final.predict(im)
features.append(outcome)


