#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2021/4/23 上午 10:49
# @Author  : H2606390
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, RMSprop
import numpy as np
from keras.utils import np_utils
from keras import regularizers
from keras.datasets import mnist
import matplotlib.pyplot as plt
from keras.utils import plot_model
import keras.backend as K

np.random.seed(1000)
# model
NB_classes = 10
optimizer = Adam()
dropout = 0.3
INPUT_SHAPE = (1,28, 28)
# 选择channels_first：返回(3,256,256)
# 选择channels_last：返回(256,256,3)
model = Sequential()
# CONN->RELU->POOL
model.add(Conv2D(20, kernel_size=5, input_shape=INPUT_SHAPE, padding='same',data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))
# CONN->RELU->POOL
model.add(Conv2D(50, kernel_size=5, padding='same',data_format='channels_first'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), data_format='channels_first'))

# Flatten -> Dense ->RELU
# Flatten层用来将输入“压平”，把多为的输入一维化，作为卷积层待全连接层的过渡
model.add(Flatten())
model.add(Dense(500))
model.add(Activation('relu'))

# Dense->softmax
model.add(Dense(NB_classes))
model.add(Activation('softmax'))
model.summary()
# 损失函数、目标函数、评价函数
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
plot_model(model, to_file='./2-Lenet-model_1.png', show_shapes=True)

# data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# we need a 60K x [1 x 28 x 28] shape as input to the CONVNET
X_train = X_train[:, np.newaxis, :, :]
X_test = X_test[:, np.newaxis, :, :]
# nomalize
X_train /= 255
X_test /= 255
y_train = np_utils.to_categorical(y_train, NB_classes)
y_test = np_utils.to_categorical(y_test, NB_classes)

# train
batch_size = 128
NB_epoch = 20
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=NB_epoch, validation_split=0.2, verbose=1)
score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print('score:', score[1], '    accuracy:', score[1])
print('history:', history.history.keys())

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('lenet_acc.png')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('lenet_loss.png')
plt.show()
