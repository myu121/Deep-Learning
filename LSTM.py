from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Input
from keras.utils import to_categorical
from keras.optimizers import *
import numpy as np
from keras.layers import LSTM
from keras import optimizers
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train_new = []
for sample in x_train:
    a = np.zeros((3, 1024))
    p = 0
    for i in range(len(sample)):
        for j in range(len(sample[0])):
            a[0][p] = sample[i][j][0]
            a[1][p] = sample[i][j][1]
            a[2][p] = sample[i][j][2]
            p += 1
    x_train_new.append(a)
x_test_new = []
for sample in x_test:
    a = np.zeros((3, 1024))
    p = 0
    for i in range(len(sample)):
        for j in range(len(sample[0])):
            a[0][p] = sample[i][j][0]
            a[1][p] = sample[i][j][1]
            a[2][p] = sample[i][j][2]
            p += 1
    x_test_new.append(a)
    
y_train_new = to_categorical(y_train, 10)
y_test_new = to_categorical(y_test, 10)
x_train_new = np.asarray(x_train_new)
x_test_new = np.asarray(x_test_new)


model = Sequential()
model.add(LSTM(128, input_shape=(None, 1024)))
model.add(Dense(10, activation='softmax'))

sgd = optimizers.SGD(lr=0.001)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train_new, y_train_new, batch_size=1024, epochs=10)
model.evaluate(x_test_new, y_test_new)
