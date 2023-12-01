#!/usr/bin/env python3


import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.text import text_to_word_sequence
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

EPOCHS = 40
BATCH_SIZE = 256
INPUT_FILE_NAME = 'frankenstein.txt'
WINDOW_LENGTH = 40
WINDOW_STEP = 3
PREDICT_LENGTH = 3
MAX_WORDS = 10000
EMBEDDING_WIDTH = 100

file = open(INPUT_FILE_NAME, 'r', encoding='utf-8')
text = file.read()
file.close()

text = text_to_word_sequence(text)

fragments = []
targets = []


for i in range(0, len(text) - WINDOW_LENGTH, WINDOW_STEP):
    fragments.append(text[i:i+WINDOW_LENGTH])
    targets.append(text[i+WINDOW_LENGTH])

# oov - out of vocabulary

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='UNK')
tokenizer.fit_on_texts(text)
framents_indexed = tokenizer.texts_to_sequences(fragments)
target_indexed = tokenizer.texts_to_sequences(targets)


X = np.array(framents_indexed, dtype=np.int)
y = np.zeros((len(target_indexed), MAX_WORDS))

for i, target_index in enumerate(target_indexed):
    y[i, target_indexed] = 1

training_model = Sequential()
training_model.add(Embedding(output_dim=EMBEDDING_WIDTH,
                   input_dim=MAX_WORDS, mask_zero=True, input_length=None))
training_model.add(LSTM(128, return_sequences=True,
                   dropout=0.2, recurrent_dropout=0.2))
training_model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
training_model.add(Dense(128, activation='relu'))
training_model.add(Dense(MAX_WORDS, activation='softmax'))
training_model.compile(loss='categorical_crossentropy', optimizer='adam')
training_model.summary()
history = training_model.fit(X, y, validation_split=0.05,
                             batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2, shuffle=True)


inference_model = Sequential()
inference_model.add(Embedding(output_dim=EMBEDDING_WIDTH,
                    input_dim=MAX_WORDS, mask_zero=True, batch_input_shape=(1,1)))
inference_model.add(LSTM(128, return_sequences=True,
                    dropout=0.2, recurrent_dropout=0.2, stateful=True))
inference_model.add(
    LSTM(128, dropout=0.2, recurrent_dropout=0.2, stateful=True))
inference_model.add(Dense(128, activation='relu'))
inference_model.add(Dense(MAX_WORDS, activation='softmax'))
weights = training_model.get_weights()
inference_model.set_weights(weights)


first_words = ['i', 'saw']
first_words_indexed = tokenizer.texts_to_sequences(first_words)
inference_model.reset_states()
predicted_string = ''


for i, word_index in enumerate(first_words_indexed):
    x = np.zeros((1, 1), dtype=np.int)
    x[0][0] = word_index[0]
    predicted_string += first_words[i]
    predicted_string += ' '
    y_predict = inference_model.predict(x, verbose=0)[0]

for i in range(PREDICT_LENGTH):
    next_word_index = np.argmax(y_predict)
    word = tokenizer.sequences_to_texts([[next_word_index]])
    x[0][0] = next_word_index
    predicted_string += word[0]
    predicted_string += ' '
    y_predict = inference_model.predict(x, verbose=0)[0]

print(predicted_string)