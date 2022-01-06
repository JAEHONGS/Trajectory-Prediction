# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 23:49:54 2021

@author: user
"""

from pytictoc import TicToc

t = TicToc()

t.tic()

import numpy as np

import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import train_test_split

file_name = 'batch_train_data1'
file_name1 = 'batch_label_data1'

cnn_data = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(file_name))
cnn_data_label = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(file_name1))

from sklearn.preprocessing import MinMaxScaler

scalers = {}

for i in range(cnn_data.shape[0]):
    for j in range(cnn_data.shape[3]):
        scalers[j] = MinMaxScaler()
        cnn_data[i, :,:, j] = scalers[j].fit_transform(cnn_data[i, :, :, j])

for i in range(cnn_data_label.shape[0]):
    scalers[i] = MinMaxScaler()
    cnn_data_label[i, :,:] = scalers[i].fit_transform(cnn_data_label[i, :, :])

# cnn_data[:, :, :, 0:2] = cnn_data[:, :, :, 0:2] / 255
# cnn_data_label = cnn_data_label / 255

CL_train, CL_test, CL_train_label, CL_test_label = train_test_split(cnn_data, cnn_data_label, test_size = 0.2, shuffle = True)

map_img_frame_ = (310,1000,4)

input_data_1 = layers.Input(shape=map_img_frame_, name = 'map_img_')

inner_cnn_rnn_ = layers.Conv2D(1,(3, 3), padding = 'same', name = 'conv1', kernel_initializer='he_normal')(input_data_1)
# inner_cnn_rnn_ = layers.Conv2D(1,(10,10), padding = 'same', name = 'conv1', kernel_initializer='he_normal')(input_data_1)
inner_cnn_rnn_ = layers.BatchNormalization()(inner_cnn_rnn_)
inner_cnn_rnn_ = layers.Activation('relu')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.MaxPooling2D(pool_size=(2,2), name='max1')(inner_cnn_rnn_)

inner_cnn_rnn_ = layers.Conv2D(1,(3, 3), padding = 'same', name = 'conv2', kernel_initializer='he_normal')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.Conv2D(1,(10,10), padding = 'same', name = 'conv2', kernel_initializer='he_normal')(input_data_1)
# inner_cnn_rnn_ = layers.BatchNormalization()(inner_cnn_rnn_)
inner_cnn_rnn_ = layers.Activation('relu')(inner_cnn_rnn_)

##시간순으로 학습
# inner_cnn_rnn_ = layers.Permute(dims=(3, 2, 1))(inner_cnn_rnn_)
inner_cnn_rnn_ = layers.Reshape(target_shape=((310,1000)), name='reshape')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.Reshape(target_shape=((500,154)), name='reshape')(inner_cnn_rnn_)


inner_cnn_rnn_ = layers.LSTM(64, return_sequences = True, name='lstm1')(inner_cnn_rnn_)
inner_cnn_rnn_ = layers.LSTM(64, return_sequences = True, name='lstm2')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.LSTM(64, name='lstm2')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='lstm2')(inner_cnn_rnn_)

## 64 STEPS을 예측
inner_cnn_rnn_ = layers.Dense(1000, activation='relu',kernel_initializer='he_normal',name='dense1')(inner_cnn_rnn_)

modelCR = Model(inputs=input_data_1, outputs=inner_cnn_rnn_)

# ============================================================================

# result = layers.concatenate([modelCR.output, y.output])

# result_rnn_layer_ = layers.LSTM(128, return_sequences = True, name='lstm5')(result)
# result_rnn_layer_ = layers.Bidirectional(layers.LSTM(128, return_sequences=True), name='lstm6')(result_rnn_layer_)
# z = layers.Dense(64, activation="relu")(result_rnn_layer_)

# modelfinal = Model(inputs=[modelCR.input, y.input], outputs=z)

# modelfinal.summary()
# modelCR.summary()

# modelfinal.compile(
#     loss=keras.losses.binary_crossentropy,
#     # loss=keras.losses.MeanSquaredError(),
#     optimizer=keras.optimizers.Adam(),
# )

modelCR.compile(
    #loss=keras.losses.binary_crossentropy,
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'],
)

early_stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=50)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10)
# mc = keras.callbacks.ModelCheckpoint('./CNN_LSTM/Model/CNN_LSTM/{epoch:02d}-{val_accuracy:.5f}_100_best.h5', monitor='accuracy', save_best_only=True)

history = modelCR.fit(
    CL_train,
    CL_train_label,
    batch_size=2,
    epochs=500,
    validation_split=0.2,
    callbacks=[early_stopping],
    # callbacks=[early_stopping, reduce_lr],
    # callbacks=[mc],
)

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

modelCR.save('./CNN_LSTM/Model/CNN_LSTM/CNN_LSTM_model_100_EP500_BS2_new_structure(3).h5')

modelCR.summary()

modelCR.evaluate(CL_test, CL_test_label)

t.toc()
