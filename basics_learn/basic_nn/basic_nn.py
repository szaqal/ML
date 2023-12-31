#!/usr/bin/env python3
import numpy as np


def show_learning():
    print('Current weights:')
    for i, w in enumerate(neurons):
        print(f'Neuron:: {i} w[0]={w[0]:5.2f} w[1]={w[1]:5.2f} w[2]={w[2]:5.2f}')



#
# XOR function learning
#
np.random.seed(3)
LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3]


x_train = [
    np.array([1.0, -1.0, -1.0]),
    np.array([1.0, -1.0, 1.0]),
    np.array([1.0, 1.0, -1.0]),
    np.array([1.0, 1.0, 1.0])
]
y_train = [0.0, 1.0, 1.0, 0.0]  # Ground truth


def neuron_w(input_count):
    wights = np.zeros(input_count+1)  # array([0., 0.])
    for i in range(1, (input_count+1)):
        # Random weights to start with
        wights[i] = np.random.uniform(-1.0, 1.0)
    return wights


neuron_input_count = 2

neurons = [
    neuron_w(neuron_input_count), #first layer
    neuron_w(neuron_input_count), #first layer
    neuron_w(neuron_input_count)  #second layer
]
neuron_outputs = [0, 0, 0]
neuron_errors = [0, 0, 0]


def forward_pass(x):
    global neuron_outputs
    neuron_outputs[0] = np.tanh(np.dot(neurons[0], x)) #tanh activation
    neuron_outputs[1] = np.tanh(np.dot(neurons[1], x)) #tanh activation

    # n2 is result neuron single with two inputs
    n2_inputs = np.array([1.0, neuron_outputs[0], neuron_outputs[1]])  # 1.0 bias
    z2 = np.dot(neurons[2], n2_inputs)
    neuron_outputs[2] = 1.0 / (1.0 + np.exp(-z2))  #sigmoid activation


def backward_pass(y_truth):
    global neuron_errors
    error_prime = -(y_truth - neuron_outputs[2])  # Compute error 

    derivative = neuron_outputs[2] * (1.0 - neuron_outputs[2])
    neuron_errors[2] = error_prime * derivative

    derivative = 1.0 - neuron_outputs[0]**2
    neuron_errors[0] = neurons[2][1] * neuron_errors[2] * derivative

    derivative = 1.0 - neuron_outputs[1]**2
    neuron_errors[1] = neurons[2][2] * neuron_errors[2] * derivative


def adjust_weigths(x):
    global neurons
    neurons[0] -= (x * LEARNING_RATE * neuron_errors[0])
    neurons[1] -= (x * LEARNING_RATE * neuron_errors[1])

    n2_inputs = np.array([1.0, neuron_outputs[0], neuron_outputs[1]])  # 1.0 bias
    neurons[2] -= (n2_inputs * LEARNING_RATE * neuron_errors[2])


if __name__ == '__main__':
    all_correct = False
    while not all_correct:
        all_correct = True
        np.random.shuffle(index_list)

        # Adjust
        for i in index_list:
            forward_pass(x_train[i])
            backward_pass(y_train[i])
            adjust_weigths(x_train[i])
            show_learning()

        # Evaluate
        for i in range(len(x_train)):

            forward_pass(x_train[i])

            print(f'x1 = {x_train[i][1]:4.1f} x2 = {x_train[i][2]:4.1f}, y = {neuron_outputs[2]:.4f}')

            # neuron output mismatch
            should_activate_mismatch = y_train[i] < 0.5 and neuron_outputs[2] >= 0.5
            should_not_activate_mistmatch = y_train[i] >= 0.5 and neuron_outputs[2] < 0.5

            if should_activate_mismatch or should_not_activate_mistmatch:
                all_correct = False
