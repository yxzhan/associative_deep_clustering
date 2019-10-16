#!/usr/bin/env python
# coding: utf-8

# # Basic classification: Classify images of Waste Material

# In[23]:


from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from importlib import import_module

print(tf.__version__)


# ## Import the Waste Material 100 dataset

# In[24]:


DATADIR = '/home/yanxiang.zyx/associative_deep_clustering/semisup/data/npy/'

dataset_tools = import_module('tools.material')

train_images, test_images, train_labels,  test_labels = dataset_tools.get_data()

# train_labels = train_labels - 1
# test_labels = test_labels - 1

class_names = ['Cardboard', 'Pamphlet', 'Empty', 'Plastic Foil', 'Shredded Paper']

NUM_LABELS = dataset_tools.NUM_LABELS
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE

train_images = train_images / 255.0

test_images = test_images / 255.0

def get_model(is_alex):
    if is_alex != True:
        return keras.Sequential([
            keras.layers.Flatten(input_shape=IMAGE_SHAPE),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dense(NUM_LABELS, activation='softmax')
        ])
    else:
        model = keras.Sequential()
        # 1st Convolutional Layer
        model.add(Conv2D(filters=96, input_shape=IMAGE_SHAPE, kernel_size=(11,11), strides=(4,4), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # 2nd Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # 3rd Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))

        # 4th Convolutional Layer
        model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))

        # 5th Convolutional Layer
        model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
        model.add(Activation('relu'))
        # Max Pooling
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))

        # Passing it to a Fully Connected layer
        model.add(Flatten())
        # 1st Fully Connected Layer
        model.add(Dense(1024, input_shape=(IMAGE_SHAPE[0]*IMAGE_SHAPE[1]*IMAGE_SHAPE[2],)))
        model.add(Activation('relu'))
        # Add Dropout to prevent overfitting
        model.add(Dropout(0.4))

        # 2nd Fully Connected Layer
        model.add(Dense(32))
        model.add(Activation('relu'))
        # Add Dropout
        model.add(Dropout(0.4))

        # 3rd Fully Connected Layer
        # model.add(Dense(1000))
        # model.add(Activation('relu'))
        # Add Dropout
        # model.add(Dropout(0.4))

        # Output Layer
        model.add(Dense(5))
        model.add(Activation('softmax'))
        return model

model = get_model(True)

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


print('training start')
model.fit(train_images, train_labels, epochs=50, shuffle=True)


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
