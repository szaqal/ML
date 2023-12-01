#!/usr/bin/env python3
# https://towardsai.net/p/machine-learning/linear-regression-complete-derivation-with-mathematics-explained

import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


url = 'https://raw.githubusercontent.com/AshishJangra27/Machine-Learning-with-Python-GFG/main/Linear%20Regression/data_for_lr.csv'
data = pd.read_csv(url)
data = data.dropna()

x_train = np.array(data.x[0:500]).reshape(500, 1)
y_train = np.array(data.y[0:500]).reshape(500, 1)

epochs = 10
lr = 0.0001


parameters = {}

m = np.random.uniform(0, 1) * -1
c = np.random.uniform(0, 1) * -1



# print(predictions(10,0,[1,2,3]))
def predict(x_train):
    global m
    global c
    pred = np.multiply(m, x_train) + c
    #pred = [m*i+c for i in x_train]
    #print(f'Predictions {pred}')    
    return pred


#print(mse([10,20,30], [10,11,20]))
def mse(predictions, truth):

    cost = np.mean((truth - predictions) ** 2)
    return cost

    #sum = 0
    #N = len(predictions)
    #for i in range(N):
    #    sum += (predictions[i] - truth[i])**2
    #return sum/N


def derivatives(x_train, y_train, predictions):

    df = (y_train - predictions) * -1
    dm = np.mean(np.multiply(x_train, df))
    dc = np.mean(df)
    return dm, dc


 #   df = []
 #   N = len(y_train)
 #   for i in range(N):
 #       df.append(y_train[i] - predictions[i])
 #   
 #   z = []
 #   for i in range(N):
 #       z.append(x_train[i]* df[i])
 #   
 #   dm = sum(z)/N
 #   dc = sum(df)/N
 #   print(f'dm:{dm}, dc:{dc}')
 #   return dm, dc




def update_params(dm, dc):
    global m
    global c
    m = m - lr * dm
    c = c - lr * dc
    #print(f'm: {m} c: {c}')
    return m, c





for i in range(epochs):
    predictions = predict(x_train)
    loss = mse(predictions, y_train)
    dm, dc = derivatives(x_train, y_train, predictions)
    update_params(dm, dc)

    print(f'{i} - {loss}')

print(f'm: {m} c: {c}')
    


#print(m, b)
x = np.linspace(0, 100)
fig, ax = plt.subplots()
ax.plot(x_train, y_train, 'o')
ax.plot(x, m*x+c)


ax.legend()
plt.show()
