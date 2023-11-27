#!/usr/bin/env python3

import numpy as np

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 3
BATCH_SIZE = 256
INPUT_FILE_NAME = 'frankenstein.txt'

# Controls training examples split
WINDOW_LENGTH = 40
WINDOW_STEP = 3


BEAM_SIZE = 8
NUM_LETTERS = 11
MAX_LENGTH = 50

file = open(INPUT_FILE_NAME, 'r', encoding='utf-8')
text = file.read()
file.close()

text = text.lower()
text = text.replace('\n', ' ')
text = text.replace('  ',  ' ')

unique_chars = list(set(text))
char_to_index = dict((ch, index) for index, ch in enumerate(unique_chars))
index_to_char = dict((index, ch) for index, ch in enumerate(unique_chars))

# One-host encoded length
encoding_width = len(char_to_index)


fragments = []
targets = []
for i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
    fragments.append(text[i: i + WINDOW_LENGTH])
    targets.append(text[i + WINDOW_LENGTH])


X = np.zeros((len(fragments), WINDOW_LENGTH, encoding_width))
y = np.zeros((len(fragments), encoding_width))

for i, fragment in enumerate(fragments):
    for j, char in enumerate(fragment):
        X[i, j, char_to_index[char]] = 1
    target_char = targets[i]
    y[i, char_to_index[target_char]] = 1

model = Sequential()
model.add(LSTM(128, return_sequences=True, dropout=0.2,
          recurrent_dropout=0.2, input_shape=(None, encoding_width)))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(encoding_width, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
model.fit(X, y, validation_split=0.05, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, shuffle=True)

letters = 'the body '
one_hots = []
for i , char in enumerate(letters):
    x = np.zeros(encoding_width)
    x[char_to_index[char]] = 1
    one_hots.append(x)

beams = [(np.log(1.0), letters, one_hots)]

for i in range(NUM_LETTERS):
    minibatch_list = []
    for triple in beams:
        minibatch_list.append(triple[2])
    minibatch = np.array(minibatch_list)
    y_predict = model.predict(minibatch, verbose=0)
    new_beams = []
    for j, softmax_vec in enumerate(y_predict):
        triple = beams[j]
        for k in range(BEAM_SIZE):
            char_index = np.argmax(softmax_vec)
            new_prob = triple[0] + np.log(softmax_vec[char_index])
            new_letters = triple[1] + index_to_char[char_index]
            x = np.zeros(encoding_width)
            x[char_index] = 1
            new_one_hots = triple[2].copy()
            new_one_hots.append(x)
            new_beams.append((new_prob, new_letters, new_one_hots))
            softmax_vec[char_index] = 0
    new_beams.sort(key=lambda tup: tup[0], reverse=True)
    beams = new_beams[0:BEAM_SIZE]

for item in beams:
    print(item[1])
