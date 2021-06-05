
import tensorflow as tf
import keras.backend as K
from data_prep import *

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

    tf.keras.layers.Dense(1024),
    tf.keras.layers.LeakyReLU(alpha=0.5),

    tf.keras.layers.Dense(512),
    tf.keras.layers.LeakyReLU(alpha=0.4),

    tf.keras.layers.Dense(256),
    tf.keras.layers.LeakyReLU(alpha=0.3),
    tf.keras.layers.Dense(2, activation='sigmoid')
])


# Настраиваем гиперпараметры нейронной сети
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Начинаем обучение
history = model.fit(train_generator,
                      validation_data=valik,
                      steps_per_epoch=5,
                      epochs=10,
                      validation_steps=3,
                      verbose=2)

model.save('save_115001.h5')





