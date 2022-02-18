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


train_file_name = 'batch_train_data1'
train_label_file_name = 'batch_train_label_data1'
test_file_name = 'batch_test_data1'
test_label_file_name = 'batch_test_label_data1'

CL_train = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(train_file_name))
CL_train_label = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/train/{}.npy'.format(train_label_file_name))

CL_test = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/test/{}.npy'.format(test_file_name))
CL_test_label = np.load('C:/Users/user/.spyder-py3/CNN_LSTM/data/test/{}.npy'.format(test_label_file_name))



map_img_frame_ = (310,1000,4)

input_data_1 = layers.Input(shape=map_img_frame_, name = 'map_img_')

inner_cnn_rnn_ = layers.Conv2D(8,(2, 2), padding = 'same', name = 'conv1')(input_data_1)
# inner_cnn_rnn_ = layers.BatchNormalization()(inner_cnn_rnn_)
inner_cnn_rnn_ = layers.Activation('relu', name = 'act1')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding ='same', name='max1')(inner_cnn_rnn_)

inner_cnn_rnn_ = layers.Conv2D(4,(2, 2), padding = 'same', name = 'conv2')(inner_cnn_rnn_)
inner_cnn_rnn_ = layers.Activation('relu', name = 'act2')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding ='same', name='max2')(inner_cnn_rnn_)

inner_cnn_rnn_ = layers.Conv2D(1,(2, 2), padding = 'same', name = 'conv3')(inner_cnn_rnn_)
inner_cnn_rnn_ = layers.Activation('relu', name = 'act3')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding ='same', name='max3')(inner_cnn_rnn_)

# inner_cnn_rnn_ = layers.Conv2D(2,(2, 2), padding = 'same', name = 'conv4')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.Activation('relu', name = 'act4')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding ='same', name='max4')(inner_cnn_rnn_)

# inner_cnn_rnn_ = layers.Conv2D(1,(2, 2), padding = 'same', name = 'conv5')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.Activation('relu', name = 'act5')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding ='same', name='max5')(inner_cnn_rnn_)

##시간순으로 학습
# # inner_cnn_rnn_ = layers.Permute(dims=(3, 2, 1))(inner_cnn_rnn_)
inner_cnn_rnn_ = layers.Reshape(target_shape=((1000,310)), name='reshape1')(inner_cnn_rnn_)
# inner_cnn_rnn_ = layers.Reshape(target_shape=((310,1000)), name='reshape1')(inner_cnn_rnn_)

inner_cnn_rnn_ = layers.LSTM(64, return_sequences = True, name='lstm1')(inner_cnn_rnn_)
inner_cnn_rnn_ = layers.Bidirectional(layers.LSTM(64, return_sequences=True), name='lstm2')(inner_cnn_rnn_)
inner_cnn_rnn_ = layers.Dense(310, activation='relu', kernel_initializer='he_normal', name='dense1')(inner_cnn_rnn_)
inner_cnn_rnn_ = layers.Reshape(target_shape=((310, 1000)), name='reshape2')(inner_cnn_rnn_)

modelCR = Model(inputs=input_data_1, outputs=inner_cnn_rnn_)

# ============================================================================

modelCR.compile(
    loss=keras.losses.MeanSquaredError(),
    optimizer=keras.optimizers.Adam(),
    metrics=['accuracy'],
)

epoch = 1500
batch_size = 8
st = '(3)'

checkpoint_path = './CNN_LSTM/Model/CNN_LSTM/best/save/%d_EP%d_BS%d%s_best_{epoch:02d}-{loss:.5f}.h5'%(len(CL_train), epoch, batch_size, st)

early_stopping = keras.callbacks.EarlyStopping(monitor="loss", patience=30)
# reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=10)

mc = keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_path,
    monitor='loss',
    save_best_only=True
)

history = modelCR.fit(
    CL_train,
    CL_train_label,
    batch_size=batch_size,
    epochs=epoch,
    validation_split=0.3,
    shuffle=True,
    # callbacks=[early_stopping],
    # callbacks=[early_stopping, mc],
    # callbacks=[early_stopping, reduce_lr],
    callbacks=[mc],
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

modelCR.summary()
modelCR.evaluate(CL_test, CL_test_label)

modelCR.save('./CNN_LSTM/Model/CNN_LSTM/CNN_LSTM_model_%d_EP%d_BS%d%s.h5'%(len(CL_train), epoch, batch_size, st))
plot_model(modelCR, to_file='./CNN_LSTM/Model/CNN_LSTM/model_plot/plot_%d_EP%d_BS%d%s.h5.png'%(len(CL_train), epoch, batch_size, st), show_shapes=True)
plt.savefig('./CNN_LSTM/Model/CNN_LSTM/loss_plot/%d_%d_%d%s.h5.png'%(len(CL_train), epoch, batch_size, st), dpi=300)

t.toc()





# from sklearn.preprocessing import MinMaxScaler

# scalers1 = {}
# scalers2 = {}

# for i in range(CL_train.shape[0]):
#     scalers1[i] = MinMaxScaler()
#     CL_train_label[i, :,:] = scalers1[i].fit_transform(CL_train_label[i, :, :])
    
#     for j in range(CL_train.shape[3] - 2):
#         CL_train[i, :,:, j] = scalers1[i].fit_transform(CL_train[i, :, :, j])

# for i in range(CL_test.shape[0]):
#     scalers2[i] = MinMaxScaler()
#     CL_test_label[i, :,:] = scalers2[i].fit_transform(CL_test_label[i, :, :])
    
#     for j in range(CL_train.shape[3] - 2):
#         CL_test[i, :,:, j] = scalers2[i].fit_transform(CL_test[i, :, :, j])
