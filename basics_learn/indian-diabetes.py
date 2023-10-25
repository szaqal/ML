#!/usr/bin/env python3.9

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from numpy import loadtxt

dataset = loadtxt('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', delimiter=',')

train = dataset[:,0:8]
label = dataset[:,8]

model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(train, label, epochs=300, batch_size=10)
# evaluate the keras model
_, accuracy = model.evaluate(train, label)
print('Accuracy: %.2f' % (accuracy*100))
print(f'{model.summary()}')