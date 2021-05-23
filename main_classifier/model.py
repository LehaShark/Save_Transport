#!/usr/bin/env python
# coding: utf-8

# In[1]:


# In[17]:


# In[4]:


from git.Save_Transport.AiGaf.data_prep import *


# In[47]:


import tensorflow as tf
import keras.backend as K
def weighted_binary_crossentropy(y_true, y_pred):
    weights = (tf.math.abs(y_true-1) * 59.) + 1.
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    weighted_bce = K.mean(bce * weights)
    return weighted_bce


# In[48]:

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), input_shape=(150, 150, 3)),
    tf.keras.layers.Conv2D(16, (3, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.Conv2D(32, (3, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.Conv2D(64, (3, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.Dense(2, activation='sigmoid')
])


# Настраиваем гиперпараметры нейронной сети
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Начинаем обучение
history = model.fit(train_generator,
                      validation_data=valik,
                      steps_per_epoch=24,
                      epochs=10,
                      validation_steps=3,
                      verbose=2)

model.save('save_1.h5')





