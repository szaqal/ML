#!/usr/bin/env python3

import os
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from keras import layers
from plot_training import plot_loss, plot_accuracy
import numpy as np
import logging


tf.get_logger().setLevel(logging.ERROR)
tf.random.set_seed(7)

EPOCHS=int(os.getenv("EPOCHS"))
BATCH_SIZE=int(os.getenv("BATCH_SIZE"))
LOSS_FUNCTION= os.getenv("LOSS_FUNCTION")
BIAS = os.getenv("BIAS")
KERNEL_INITIALIZER=os.getenv("KERNEL_INITIALIZER")
LEARNING_RATE=float(os.getenv("LEARNING_RATE"))
OPTIMIZER=os.getenv("OPTIMIZER")

initializer = None
if KERNEL_INITIALIZER == "random_uniform":
    initializer = keras.initializers.RandomUniform(minval=0.1, maxval=0.1)
elif KERNEL_INITIALIZER == "glorot_uniform":
    initializer = KERNEL_INITIALIZER




print("*"*30)
print(f"Initializer: {initializer}")






mnist = keras.datasets.mnist
(train_images,train_labels), (test_images, test_labels) = mnist.load_data()

mean = np.mean(train_images)
stddev = np.std(train_images)

train_images = (train_images - mean)/stddev
test_images = (test_images - mean)/stddev

train_labels = to_categorical(train_labels, num_classes=10)
test_labels = to_categorical(test_labels, num_classes=10)



strategy = tf.distribute.MirroredStrategy()
print(f'Number of devices: {strategy.num_replicas_in_sync}')

with strategy.scope():
    model = keras.Sequential([        
        layers.Flatten(input_shape=(28,28)), # redudant if data already in singel dimention array
        layers.Dense(25, activation='tanh', kernel_initializer = initializer, bias_initializer=BIAS),  #Dense fully connected
        layers.BatchNormalization(),
        layers.Dense(10, activation='sigmoid', kernel_initializer = initializer, bias_initializer=BIAS)
    ])

    opt = None
    if OPTIMIZER == "sgd":
        opt = keras.optimizers.SGD(learning_rate=LEARNING_RATE)
    elif OPTIMIZER =="adam":
        opt = OPTIMIZER

    print(f"{opt}")
    print("*"*30)
    model.compile(loss=LOSS_FUNCTION, optimizer=opt, metrics=['accuracy'])

history = model.fit(
        train_images, 
        train_labels, 
        validation_data=(test_images, test_labels), 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE, 
        verbose=2, 
        shuffle=True
)

plot_loss(history)
plot_accuracy(history)
