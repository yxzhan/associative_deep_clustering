#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from importlib import import_module
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1000)
ONE_HOT_LABEL = True


# In[2]:


def count_data(data):
    if ONE_HOT_LABEL:
        data = data.dot(np.arange(0,5)).astype('uint8')
    unique, counts = np.unique(data, return_counts=True)
    print(dict(zip(unique, counts)))


# In[3]:


# # (2) Get Data
# import tflearn.datasets.oxflower17 as oxflower17
# x, y = oxflower17.load_data(one_hot=True)

DATADIR = '/Users/yanxiang.zyx/KIPRO/associative_deep_clustering/semisup/data/npy/'

dataset_tools = import_module('tools.material')

train_images, test_images, train_labels,  test_labels = dataset_tools.get_data(one_hot=ONE_HOT_LABEL, test_size=0.1)

class_names = ['Cardboard', 'Pamphlet', 'Empty', 'Plastic Foil', 'Shredded Paper']

NUM_LABELS = dataset_tools.NUM_LABELS
IMAGE_SHAPE = dataset_tools.IMAGE_SHAPE

print(train_images.shape)

count_data(train_labels)
count_data(test_labels)


# In[4]:


# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(train_images[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[np.argmax(train_labels[i])])
# plt.show()


# In[5]:


# (3) Create a sequential model
model = Sequential()

# model.add(Dropout(0.5, input_shape=IMAGE_SHAPE))
model.add(BatchNormalization(input_shape=IMAGE_SHAPE,         axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros',        gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None,        gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))

# 1st Convolutional Layer
model.add(Conv2D(filters=96,      kernel_size=(11,11), strides=(4,4), padding='valid',     trainable=False,      activation = 'relu',      kernel_initializer='glorot_uniform'))
# Pooling 
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2),        trainable=False,         padding='valid'))

# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same',          trainable=False,          activation = 'relu',          kernel_initializer='glorot_uniform'))
# Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2),         trainable=False,         padding='valid'))

# 3rd Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same',          trainable=False,          activation = 'relu',          kernel_initializer='glorot_uniform'))

# 4th Convolutional Layer
model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='same',          activation = 'relu',          kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(momentum=0.9))

# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same',          activation = 'relu',          kernel_initializer='glorot_uniform'))
model.add(BatchNormalization(momentum=0.9))
# Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))

# Passing it to a dense layer
model.add(Flatten())

# 1st Dense Layer
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# 2nd Dense Layer
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(5))
model.add(Activation('softmax'))

model.summary()


# In[6]:


def get_lr_metric(a):
    def lr(y_true, y_pred):
        return a.lr
    return lr


# In[7]:


# (4) Compile 
loss_func = 'categorical_crossentropy' if ONE_HOT_LABEL else 'sparse_categorical_crossentropy'
adam = keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, decay=0.0001, amsgrad=False)
# model.compile(loss=loss_func, optimizer=adam, metrics=['accuracy'])
# lr_metric = get_lr_metric(adam)

model.compile(loss=loss_func, optimizer=adam, metrics=['accuracy'])
each_epoch = 10
total_epochs = 0


# In[8]:


# (5) Train
print(total_epochs)
total_epochs += each_epoch
model.fit(train_images, train_labels, epochs=each_epoch, verbose=1, validation_split=0.2, shuffle=True)


# In[9]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1)

print('\nTest loss:', test_loss)
print('\nTest accuracy:', test_acc)


# In[10]:


predictions = model.predict(test_images)


# In[11]:


def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  if ONE_HOT_LABEL:
    true_label = np.argmax(true_label)
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  if ONE_HOT_LABEL:
    true_label = np.argmax(true_label)
  plt.grid(False)
  plt.xticks(range(NUM_LABELS))
  plt.yticks([])
  thisplot = plt.bar(range(NUM_LABELS), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')


# In[12]:


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()


# In[ ]:




