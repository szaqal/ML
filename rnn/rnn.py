#!/usr/bin/env python3

import logging
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN


# How to download https://github.com/NVDLI/LDL

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 100
BATCH_SIZE = 16
TRAIN_TEST_SPLIT = 0.8
MIN = 12
FILE_NAME = "SeriesReport-202310160623.csv"


def readfile(file_name):
    file = open(file_name, 'r', encoding='utf-8')
    next(file)
    data = []
    for line in (file):
        values = line.split(',')
        data.append(float(values[1]))
    file.close()
    return np.array(data, dtype=np.float32)


sales = readfile(FILE_NAME)
months = len(sales)
split = int(months * TRAIN_TEST_SPLIT)
train_sales = sales[0:split]
test_sales = sales[split:]


# plot
# x = range(len(sales))
# plt.plot(x, sales, 'r-', label='sales')
# plt.title('Book store')
#plt.axis([0,339,0.0, 3000.0])
# plt.xlabel('Months')
# plt.ylabel('Sales')
# plt.legend()
# plt.show()


mean = np.mean(train_sales)
stddev = np.std(train_sales)
train_sales_std = (train_sales - mean) / stddev
test_sailes_std = (train_sales - mean) / stddev


train_months = len(train_sales)
train_x = np.zeros((train_months - MIN, train_months - 1, 1))
train_y = np.zeros((train_months - MIN, 1))

for i in range(0, train_months - MIN):
    train_x[i, -(i+MIN):, 0] = train_sales_std[0:i+MIN]
    train_y[i, 0] = train_sales_std[i+MIN]


test_months = len(test_sales)
test_x = np.zeros((test_months-MIN, test_months -1, 1))
test_y = np.zeros((test_months-MIN, 1))
for i in range(0, test_months - MIN):
    test_x[i, -(i+MIN):, 0] = test_sailes_std[0:i+MIN]
    test_y[i, 0] = test_sailes_std[i+MIN]


model = Sequential()
model.add(SimpleRNN(128, activation='relu', input_shape=(None, 1)))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
model.summary()

history = model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)


