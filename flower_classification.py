import keras
from enum import Enum
import csv
import tkinter as tk
from keras.callbacks import ModelCheckpoint
from tkinter import filedialog as fd
from keras import layers
from keras import models
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import losses
from keras.layers import Dropout
import matplotlib.pyplot as plt
import numpy as np
import os


def plt_modle(model_hist):
    acc = model_hist.history['acc']
    val_acc = model_hist.history['val_acc']
    loss = model_hist.history['loss']
    val_loss = model_hist.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 6));
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, color='#0984e3', marker='o', linestyle='none', label='Training Accuracy')
    plt.plot(epochs, val_acc, color='#0984e3', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, color='#eb4d4b', marker='o', linestyle='none', label='Training Loss')
    plt.plot(epochs, val_loss, color='#eb4d4b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()



def continue_train_model(model_path,epochs,train_num):
    optimizer = keras.optimizers.SGD(lr=0.0001)
    loss = keras.losses.categorical_crossentropy
    model = load_model(model_path)
    checkpoint = ModelCheckpoint('flowers_checkpoint_test', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, callbacks=callbacks_list, verbose=0)
    model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs= epochs, callbacks=callbacks_list , validation_data=valid_gen, validation_steps=v_steps)
    model.save('flowers_model_continue_with_cp2.h5')
    plt_modle(model_hist)


# Split images into Training and Validation Sets (20%)

train = ImageDataGenerator(rescale=1./255,horizontal_flip=True, shear_range=0.2, zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)

img_size = 128
batch_size = 20
t_steps = 3462/batch_size
v_steps = 861/batch_size
classes = 5
flower_path = 'flowers'
train_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode='categorical', subset='training')
valid_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical', subset='validation')




model = models.Sequential()
#CONV + MAX POOL 1
model.add(Conv2D(32,kernel_size=(4,4),input_shape=(128,128,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))

#CONV + MAX POOL 2
model.add(Conv2D(64,kernel_size=(4,4),input_shape=(128,128,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
#CONV + MAX POOL 3
model.add(Conv2D(128,kernel_size=(4,4),input_shape=(128,128,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
#CONV + MAX POOL 4
model.add(Conv2D(256,kernel_size=(4,4),input_shape=(128,128,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
#CONV + MAX POOL 5
model.add(Conv2D(256,kernel_size=(4,4),input_shape=(128,128,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
# model.add(Dropout(0.5))
#CONV + MAX POOL 5
model.add(Conv2D(128,kernel_size=(4,4),input_shape=(128,128,3),strides=(1,1),padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same'))
#Avoiding overfit
model.add(Dropout(0.5))
model.add(Flatten())
model.add(layers.Dense(2048, activation='relu'))
model.add(layers.Dense(classes, activation='softmax'))

optimizer = keras.optimizers.SGD(lr=0.0001)
loss = keras.losses.categorical_crossentropy

model.compile(loss= loss, optimizer=optimizer, metrics=['accuracy'])
model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs= 1 , validation_data=valid_gen, validation_steps=v_steps)
model.save('flowers_model.h5')
plt_modle(model_hist)







