# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 17:22:04 2021

@author: user
"""

import numpy as np

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

train_file_name = 'batch_train_data2'
label_file_name = 'batch_label_data2'

vel_train = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(train_file_name))
vel_label = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(label_file_name))

vel_train[:,:,0:2] = vel_train[:,:,0:2] / 100
vel_label[:,:,0:2] = vel_label[:,:,0:2] / 100

x_train, x_test, y_train, y_test = train_test_split(vel_train, vel_label, test_size = 0.2, shuffle = True)

vel_data_seq_ = (300,4)
input_data_2 = layers.Input(shape=vel_data_seq_, name = 'input2')

rnn_layer_ = layers.LSTM(64, return_sequences = True, name='lstm3')(input_data_2)
rnn_layer_ = layers.BatchNormalization()(rnn_layer_)
rnn_layer_ = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='lstm4')(rnn_layer_)
# rnn_layer_ = layers.BatchNormalization()(rnn_layer_)
# rnn_layer_ = layers.Dense(64, activation='relu',kernel_initializer='he_normal',name='dense3')(rnn_layer_)
rnn_layer_ = layers.Dense(4, activation='relu',kernel_initializer='he_normal',name='dense4')(rnn_layer_)

modelR = Model(inputs=input_data_2, outputs=rnn_layer_)

modelR.summary()

modelR.compile(
    #loss=keras.losses.binary_crossentropy, 
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=50)  # monitor='val_loss'
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=100)
# mc = keras.callbacks.ModelCheckpoint('./CNN_LSTM/Model/LSTM/{epoch:02d}-{val_accuracy:.5f}_100_best.h5', monitor='val_accuracy', save_best_only=True)

history = modelR.fit(
    x_train,
    y_train,
    batch_size=4,
    epochs=100,
    validation_split=0.2,
    # validation_data=(x_val, y_val),
    # callbacks=[reduce_lr],
    # callbacks=[early_stopping, reduce_lr],
    # callbacks=[mc],
)

#모델 시각화
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], 'b--', label='loss')
plt.plot(history.history['val_loss'], 'r:', label='val loss')
plt.xlabel('Epochs')
plt.grid()
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], 'b--', label='accuracy')
plt.plot(history.history['val_accuracy'], 'r:', label='val accuracy')
plt.xlabel('Epochs')
plt.grid()
plt.legend()

modelR.evaluate(x_test, y_test)

modelR.save('LSTM_model_100.h5')

#plt.savefig('시각화.png', dpi=300)