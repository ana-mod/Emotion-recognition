import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation,Flatten,BatchNormalization
from keras.layers import Conv2D,MaxPooling2D
import numpy as np
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import pandas as pd
import os
from keras_vggface.vggface import VGGFace
from keras.layers import Activation, Convolution2D, Dropout, Conv2D
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.layers import Flatten
from keras.models import Model
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import SeparableConv2D
from keras import layers
from keras.regularizers import l2

num_class = 7
my_model = Sequential()
my_model.add(VGGFace(include_top=False, model='resnet50',  pooling='avg', input_shape = (224,224,3)))
my_model.add(Dense(7, activation='softmax'))
for layer in my_model.layers:
    layer.trainable='False'

my_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
                					rescale=1./255,
                rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2)
            
train_generator = train_datagen.flow_from_directory(
                'train',
                target_size=(224, 224),
                batch_size=32,
                color_mode='rgb',
                class_mode='categorical',
                shuffle=True)

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
                'validation',
                target_size=(224, 224),
                batch_size=32,
                color_mode='rgb',
                class_mode='categorical',
                )  

checkpoint = ModelCheckpoint('vggface.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             verbose=1)

earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=7,
                          verbose=1,
                          restore_best_weights=True
                          )

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.2,
                              patience=3,
                              verbose=1,
                              min_delta=0.0001)

callbacks = [earlystop,checkpoint,reduce_lr]
nb_train_samples = 28789
nb_validation_samples = 3589

history = my_model.fit_generator(train_generator, steps_per_epoch=28789//32, epochs=10, callbacks=callbacks, validation_data=validation_generator, validation_steps=3589//32)
