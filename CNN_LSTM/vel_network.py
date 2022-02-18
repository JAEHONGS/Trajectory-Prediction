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

import matplotlib.pyplot as plt



train_file_name = 'batch_train_data2'
train_label_file_name = 'batch_train_label_data2'
test_file_name = 'batch_test_data2'
test_label_file_name = 'batch_test_label_data2'

vel_train = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(train_file_name))
vel_train_label = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(train_label_file_name))

vel_test = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/test/{}.npy'.format(test_file_name))
vel_test_label = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/test/{}.npy'.format(test_label_file_name))

vel_train[:,:,0:2] = vel_train[:,:,0:2] / 500
vel_train_label[:,:,0:2] = vel_train_label[:,:,0:2] / 500
vel_test[:,:,0:2] = vel_test[:,:,0:2] / 500
vel_test_label[:,:,0:2] = vel_test_label[:,:,0:2] / 500


vel_data_seq_ = (200,4)
input_data_2 = layers.Input(shape=vel_data_seq_, name = 'input2')

rnn_layer_ = layers.LSTM(64, return_sequences = True, name='lstm1')(input_data_2)
rnn_layer_ = layers.Bidirectional(layers.LSTM(64, return_sequences = True), name='lstm2')(rnn_layer_)
# rnn_layer_ = layers.BatchNormalization()(rnn_layer_)
rnn_layer_ = layers.Dense(4, activation='relu',kernel_initializer='he_normal',name='dense1')(rnn_layer_)
# rnn_layer_ = layers.Reshape(target_shape=(20, 4), name='reshape1')(rnn_layer_)

modelR = Model(inputs=input_data_2, outputs=rnn_layer_)

modelR.compile(
    #loss=keras.losses.binary_crossentropy, 
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'],
)

modelR.summary() 

epoch = 1000
batch_size = 4
st = '(5)'

checkpoint_path = './CNN_LSTM/Model/LSTM/best/save/%d_EP%d_BS%d%s_best_{epoch:02d}-{val_loss:.5f}.h5'%(len(vel_train), epoch, batch_size, st)

early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=30)  # monitor='val_loss'
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=100)
mc = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    monitor='val_loss',
    save_best_only=True
)

history = modelR.fit(
    vel_train,
    vel_train_label,
    batch_size = batch_size,
    epochs = epoch,
    validation_split=0.3,
    # callbacks=[reduce_lr],
    # callbacks=[early_stopping]
    # callbacks=[early_stopping, mc],
    # callbacks=[early_stopping, reduce_lr],
    callbacks=[mc],
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

modelR.summary()           
modelR.evaluate(vel_test, vel_test_label)

modelR.save('./CNN_LSTM/Model/LSTM/LSTM_model_%d_EP%d_BS%d%s.h5'%(len(vel_train), epoch, batch_size, st))
plot_model(modelR, to_file='./CNN_LSTM/Model/LSTM/model_plot/plot_%d_EP%d_BS%d%s.h5.png'%(len(vel_train), epoch, batch_size, st), show_shapes=True)
plt.savefig('./CNN_LSTM/Model/LSTM/loss_plot/%d_%d_%d%s.h5.png'%(len(vel_train), epoch, batch_size, st), dpi=300)


# from sklearn.preprocessing import MinMaxScaler

# scalers = {}

# for i in range(vel_train.shape[0]):
#     scalers[i] = MinMaxScaler()
#     vel_train[i, :, 0:2] = scalers[i].fit_transform(vel_train[i, :, 0:2])
#     vel_label[i, :, 0:2] = scalers[i].fit_transform(vel_label[i, :, 0:2])
