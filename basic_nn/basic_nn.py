#!/usr/bin/env python
import numpy as np


def show_learning():
    print('Current weights:')
    for i, w in enumerate(n_w):
        print('neuron ', i, ': w0=',
              '%5.2f' % w[0], ', w1=', '%5.2f' % w[1], ', w2=', '%5.2f' % w[2])


# XOR function
np.random.seed(3)
LEARNING_RATE = 0.1
index_list = [0, 1, 2, 3]

x_train = [
    np.array([1.0, -1.0, -1.0]),
    np.array([1.0, -1.0, 1.0]),
    np.array([1.0, 1.0, -1.0]),
    np.array([1.0, 1.0, 1.0])
]

y_train = [0.0, 1.0, 1.0, 0.0]


def neuron_w(input_count):
    wights = np.zeros(input_count+1)
    for i in range(1, (input_count+1)):
        wights[i] = np.random.uniform(-1.0, 1.0)
    return wights


n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]
n_y = [0, 0, 0]
n_error = [0, 0, 0]


def forward_pass(x):
    global n_y
    n_y[0] = np.tanh(np.dot(n_w[0], x))
    n_y[1] = np.tanh(np.dot(n_w[1], x))
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])  # 1.0 bias
    z2 = np.dot(n_w[2], n2_inputs)
    n_y[2] = 1.0 / (1.0 + np.exp(-z2))


def backward_pass(y_truth):
    global n_error
    error_prime = -(y_truth - n_y[2])
    derivative = n_y[2] * (1.0 - n_y[2])
    n_error[2] = error_prime * derivative

    derivative = 1.0 - n_y[0]**2
    n_error[0] = n_w[2][1] * n_error[2] * derivative

    derivative = 1.0 - n_y[1]**2
    n_error[1] = n_w[2][2] * n_error[2] * derivative


def adjust_weigths(x):
    global n_w
    n_w[0] -= (x * LEARNING_RATE * n_error[0])
    n_w[1] -= (x * LEARNING_RATE * n_error[1])
    n2_inputs = np.array([1.0, n_y[0], n_y[1]]) # 1.0 bias
    n_w[2] -= (n2_inputs * LEARNING_RATE * n_error[2])


if __name__ == '__main__':
    all_correct = False
    while not all_correct:
        all_correct = True
        np.random.shuffle(index_list)
        for i in index_list:
            forward_pass(x_train[i])
            forward_pass(y_train[i])
            adjust_weigths(x_train[i])
            show_learning()
        for i in range(len(x_train)):
            forward_pass(x_train[i])
            print('x1 = ', '%4.1f' % x_train[i][1], ', x2 = ', '%4.1f' % x_train[i][2], ', y = ', '%4.1f' % n_y[2])
            if(((y_train[i]<0.6) and (n_y[2]>=0.5)) or ((y_train[i] >=0.5) and (n_y[2]< 0.5))):
                all_correct = True



