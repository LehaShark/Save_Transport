#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
from memory_profiler import memory_usage
import os
import pandas as pd
from glob import glob
import numpy as np


# In[2]:


get_ipython().run_cell_magic('capture', '', '!apt-get install libav-tools -y')


# In[1]:


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


# In[61]:


def create_spectrogram(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    
    filename  = '/home/aigaf/Desktop/audio-classification-master/Save_Transport/AiGaf/train/Train_jpg/Train_Noise/' + name + '.jpg'
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


# In[62]:


def create_spectrogram_test(filename,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = Path('/home/aigaf/Desktop/audio-classification-master/Save_Transport/AiGaf/test/Test_jpg/Test_Noise/' + name + '.jpg')
    fig.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()    
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


# In[17]:


from pydub import AudioSegment
import math

class SplitWavAudioMubin():
    def __init__(self, folder, filename):
        self.folder = folder
        self.filename = filename
        self.filepath = folder + '/' + filename
        
        self.audio = AudioSegment.from_wav(self.filepath)
    
    def get_duration(self):
        return self.audio.duration_seconds
    
    def single_split(self, from_min, to_min, split_filename):
        t1 = from_min * 1000
        t2 = to_min * 1000
        split_audio = self.audio[t1:t2]
        split_audio.export('/home/aigaf/Desktop/GitHub/Save_Transport/dataset/train/5sec_cut/' + split_filename, format="wav")
        
    def multiple_split(self, min_per_split):
        total_mins = math.ceil(self.get_duration() / 1)
        for i in range(0, total_mins, min_per_split):
            split_fn = str(i) + '_' + self.filename
            self.single_split(i, i+min_per_split, split_fn)
            print(str(i) + ' Done')
            if i == total_mins - min_per_split:
                print('All splited successfully ' + self.filename)


# In[18]:


import os
files = os.listdir("/home/aigaf/Desktop/GitHub/Save_Transport/dataset/train/cut")

folder = '/home/aigaf/Desktop/GitHub/Save_Transport/dataset/train/cut'

for file in files:
    split_wav = SplitWavAudioMubin(folder, file)
    split_wav.multiple_split(min_per_split=1)


# In[9]:


Data_dir=np.array(glob("../AiGaf/train/Train_wav/Train_Noise/*"))
#%load_ext memory_profiler
#%memit 

i=0
for file in Data_dir[i:i+2000]:
    #Define the filename as is, "name" refers to the JPG, and is split off into the number itself. 
    filename,name = file,file.split('/')[-1].split('.')[0]
    create_spectrogram(filename,name)
gc.collect()


# In[4]:


import numpy as np
from sklearn.metrics import fbeta_score
from keras import backend as K


def fbeta(y_true, y_pred, threshold_shift=0):
    beta = 0.5

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin), axis=1) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)), axis=1)
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)), axis=1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + K.epsilon()))

y_true, y_pred = np.round(np.random.rand(100, 3)), np.round(np.random.rand(100, 3))
# ensure, that y_true has at least one 1, because sklearn's fbeta can't handle all-zeros
y_true[:, 0] += 1 - y_true.sum(axis=1).clip(0, 1)

#fbeta_keras = fbeta(K.variable(y_true), K.variable(y_pred)).eval(session=K.get_session())
fbeta_sklearn = fbeta_score(y_true, np.round(y_pred), beta=0.5, average='samples')

print('Scores are {:.3f} (sklearn)'.format(fbeta_sklearn))


# In[5]:


from keras_preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        '/home/aigaf/Desktop/audio-classification-master/Save_Transport/AiGaf/train/Train_jpg',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '/home/aigaf/Desktop/audio-classification-master/Save_Transport/AiGaf/validation/',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary')

print("Image preprocessing complete")


# In[47]:


import tensorflow as tf
import keras.backend as K
def weighted_binary_crossentropy(y_true, y_pred):
    weights = (tf.math.abs(y_true-1) * 59.) + 1.
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce


# In[48]:



from tensorflow.keras.optimizers import RMSprop
model = Sequential() 
model.add(layers.Conv2D(32, (3, 3), padding='same',
                 input_shape=(150,150,3)))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Conv2D(128, (3, 3), padding='same'))
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(128, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.5))
model.add(layers.Flatten())
model.add(layers.Dense(512))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(RMSprop(lr=0.0005, decay=1e-6),loss=weighted_binary_crossentropy,
              metrics=[fbeta],run_eagerly=True)
model.summary()


# In[53]:


#Fitting keras model, no test gen for now
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validation_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=3
)
model.evaluate_generator(generator=validation_generator, steps=STEP_SIZE_VALID)


# In[54]:


preds = model.predict(X_test)
y_pred = np.where(preds>0.5,1,0)
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.show()


# In[ ]:




