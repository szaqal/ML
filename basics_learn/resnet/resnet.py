#!/usr/bin/env python3

import numpy as np
from tensorflow import keras
from keras.applications import resnet50
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.resnet50 import decode_predictions

import matplotlib.pyplot as plt
import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)

image = load_img('./maltese-portrait.jpg', target_size=(224, 224))
image_np = img_to_array(image)
image_np = np.expand_dims(image_np, axis=0)


model = resnet50.ResNet50(weights='imagenet')
X = resnet50.preprocess_input(image_np.copy())

y = model.predict(X)
predicted_labels = decode_predictions(y)
print(f'labels = {predicted_labels}')
plt.imshow(np.uint8(image_np[0]))
plt.show()