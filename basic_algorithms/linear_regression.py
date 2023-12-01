#!/usr/bin/env python3
# https://towardsai.net/p/machine-learning/linear-regression-complete-derivation-with-mathematics-explained
# https://towardsdatascience.com/gradient-descent-from-scratch-e8b75fa986cc
# https://towardsai.net/p/machine-learning/linear-regression-complete-derivation-with-mathematics-explained

import random
import numpy as np
import matplotlib.pyplot as plt


file = open('lr.csv', 'r', encoding='utf-8')
text = file.readlines()
file.close()

# [[],[]] - format
temp = [line.strip().split(',') for line in text]
# [[] , ..]
x_train = [int(x[0]) for x in temp]
# [.. , []]
y_train = [int(y[1]) for y in temp]

epochs = 100
lr = 0.0001


parameters = {}

m = random.randint(0,3)
c = random.randint(0,3)


def predict(x_train):
    global m
    global c
    return [m*i+c for i in x_train]


def mse(predictions, truth):
    sum = 0
    N = len(predictions)
    for i in range(N):
        sum += (predictions[i] - truth[i])**2
    return sum/N


def derivatives(x_train, y_train, predictions):

    df = []
    N = len(y_train)
    for i in range(N):
        df.append((y_train[i] - predictions[i])* -1) 
   
    z = []
    for i in range(N):
        z.append(x_train[i]* df[i])
    
    dm = sum(z)/N
    dc = sum(df)/N
    print(f'dm:{dm}, dc:{dc}')
    return dm, dc


def update_params(dm, dc):
    global m
    global c
    m = m - lr * dm
    c = c - lr * dc
    return m, c


for i in range(epochs):
    predictions = predict(x_train)
    loss = mse(predictions, y_train)
    dm, dc = derivatives(x_train, y_train, predictions)
    update_params(dm, dc)
    print(f'{i} - {loss}')
    

x = np.linspace(0, 100)
fig, ax = plt.subplots()
ax.plot(x_train, y_train, 'o')
ax.plot(x, m*x+c)


ax.legend()
plt.show()
