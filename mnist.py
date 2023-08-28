#!/usr/bin/env python

import idx2numpy
import numpy as np
import timeit
import matplotlib.pyplot as plt


chart_x = []
chart_y_train = []
chart_y_test = []

np.random.seed(7)
LEARNING_RATE = 0.01
EPOCHS = 20


def show_learning(epoch_no, train_acc, test_acc):
    global chart_x
    global chart_y_train
    global chart_y_test

    print(
        f'epoch no: {epoch_no}, train_acc: {train_acc:6.4f}, test_acc: {test_acc:6.4f}')

    chart_x.append(epoch_no + 1)
    chart_y_train.append(1.0 - test_acc)
    chart_y_test.append(1.0-test_acc)


def plot_learning():
    plt.plot(chart_x, chart_y_train, 'r-', label='training error')
    plt.plot(chart_x, chart_y_test, 'b-', label='test error')
    plt.axis([0, len(chart_x), 0.0, 1.0])
    plt.xlabel('training epochs')
    plt.ylabel('error')
    plt.legend()
    plt.show()


def read_mnist():
    TRAIN_IMAGE_FILENAME = 'data/train-images-idx3-ubyte'
    TRAIN_LABEL_FILENAME = 'data/train-labels-idx1-ubyte'
    TEST_IMAGE_FILENAME = 'data/t10k-images-idx3-ubyte'
    TEST_LABEL_FILENAME = 'data/t10k-labels-idx1-ubyte'

    train_images = idx2numpy.convert_from_file(TRAIN_IMAGE_FILENAME)
    train_labels = idx2numpy.convert_from_file(TRAIN_LABEL_FILENAME)
    test_images = idx2numpy.convert_from_file(TEST_IMAGE_FILENAME)
    test_labels = idx2numpy.convert_from_file(TEST_LABEL_FILENAME)

    x_train = train_images.reshape(60000, 784)  # 28x28 x 60000 examples
    mean = np.mean(x_train)
    stddev = np.std(x_train)
    # Optional but used for performance reasonst
    x_train = (x_train - mean) / stddev

    x_test = test_images.reshape(10000, 784)
    # Optional but used for performance reasonst
    x_test = (x_test - mean) / stddev

    y_train = np.zeros((60000, 10))  # One hot encoding for 10 digits
    y_test = np.zeros((10000, 10))  # One hot encoding for 10 digits

    for i, y in enumerate(train_labels):
        y_train[i][y] = 1
    for i, y in enumerate(test_labels):
        y_test[i][y] = 1
    return x_train, y_train, x_test, y_test


def layer_w(neuron_count, input_count):
    weights = np.zeros((neuron_count, input_count + 1))
    for neuron in range(neuron_count):
        for input in range(1, (input_count+1)):
            weights[neuron][input] = np.random.uniform(-1.0, 1.0)
    return weights


hidden_layer_neuron_count = 25
hidden_layer_input_count = 784
hidden_layer_w = layer_w(hidden_layer_neuron_count, hidden_layer_input_count)
hidden_layer_y = np.zeros(hidden_layer_neuron_count)
hidden_layer_error = np.zeros(hidden_layer_neuron_count)

output_layer_neuron_count = 10
output_layer_input_count = 25
output_layer_w = layer_w(output_layer_neuron_count, output_layer_input_count)
output_layer_y = np.zeros(output_layer_neuron_count)
output_layer_error = np.zeros(output_layer_neuron_count)


def forward_pass(x):
    global hidden_layer_y
    global output_layer_y

    for i, w in enumerate(hidden_layer_w):
        z = np.dot(w, x)
        hidden_layer_y[i] = np.tanh(z)

    hidden_output_array = np.concatenate((np.array([1.0]), hidden_layer_y))

    for i, w in enumerate(output_layer_w):
        z = np.dot(w, hidden_output_array)
        output_layer_y[i] = 1.0/(1.0 + np.exp(-z))


def backward_pass(y_truth):
    global hidden_layer_error
    global output_layer_error

    for i, y in enumerate(output_layer_y):
        error_prime = -(y_truth[i]-y)
        derivative = y * (1.0 - y)  # Logistic
        output_layer_error[i] = error_prime * derivative


    for i, y in enumerate(hidden_layer_y):
        error_weights = []
        for w in output_layer_w:
            error_weights.append(w[i+1])
        error_weight_array = np.array(error_weights)
        derivative = 1.0 - y**2
        weighted_error = np.dot(error_weight_array, output_layer_error)
        hidden_layer_error[i] = weighted_error * derivative

def adjust_weights(x):
    
    global output_layer_w
    global hidden_layer_w

    for i, error in enumerate(hidden_layer_error):
        hidden_layer_w[i] -= (x * LEARNING_RATE * error)

    (hidden_output_array) = np.concatenate((np.array([1.0]), hidden_layer_y))
    for i, error in enumerate(output_layer_error):
        output_layer_w[i] -= (hidden_output_array * LEARNING_RATE * error)



x_train ,y_train, x_test, y_test = read_mnist()
index_list = list(range(len(x_train)))

for i in range(EPOCHS):
    np.random.shuffle(index_list)
    correct_training_results = 0
    

    starttime = timeit.default_timer()
    for j in index_list:        
        x = np.concatenate((np.array([1.0]), x_train[j]))
        forward_pass(x)
        if output_layer_y.argmax() == y_train[j].argmax():
            correct_training_results +=1
        backward_pass(y_train[j])
        adjust_weights(x)

    print("Duration :", timeit.default_timer() - starttime)        

    correct_test_results = 0

    for j in range(len(x_test)):
        x = np.concatenate((np.array([1.0]), x_test[j]))
        forward_pass(x)
        if output_layer_y.argmax() == y_test[j].argmax():
            correct_test_results +=1
        
    show_learning(i, correct_training_results/len(x_train), correct_test_results/len(x_test))
plot_learning()