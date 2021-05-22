#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import pandas as pd
from glob import glob
import numpy as np


from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from keras import models
from keras.layers.advanced_activations import LeakyReLU

from tensorflow.keras.optimizers import Adam
import keras.backend as K
import librosa
import librosa.display
import pylab
import matplotlib.pyplot as plt
from matplotlib import figure
import gc
from path import Path


from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255)
test_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
        '/home/timur/Documents/Projects/sound_classif/git/Save_Transport/dataset/train/.jpg',
        target_size=(150, 150),
        batch_size=20)

valik = test_datagen.flow_from_directory(
        '/home/timur/Documents/Projects/sound_classif/git/Save_Transport/dataset/test/',
        target_size=(150, 150),
        batch_size=20)
