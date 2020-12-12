# -*- coding: utf-8 -*-



import tensorflow as tf
from tensorflow import keras


import numpy as np
import matplotlib.pyplot as plt


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.ops.variables import Variable
from tensorflow.python.tools import import_pb_to_tensorboard

train_dir = 'train'
val_dir = 'val'
test_dir = 'test'

img_width, img_height = 150, 150
input_shape = (img_width, img_height, 3)

epochs = 2
batch_size = 100
nb_train_samples = 25073
nb_validation_samples = 6215
nb_test_samples = 6285


model = Sequential()
model.add(Conv2D(20, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(20, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(25, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(19))
model.add(Activation('sigmoid'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

val_generator = datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')


test_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=nb_validation_samples // batch_size)
"""
saver = tf.train.Saver()
tf.train.write_graph(session.graph_def, path_to_folder, "net.pb", False)
tf.train.write_graph(session.graph_def, path_to_folder, "net.pbtxt", True)
saver.save(session,path_to_folder+"model.ckpt")


model_json = model.to_json()
json_file = open("cvd_model.json", "w")
# Записываем архитектуру сети в файл
json_file.write(model_json)
json_file.close()
"""
model.save('shazam.rb')