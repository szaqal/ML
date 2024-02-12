#!/usr/bin/env python3

import numpy as np

x = np.eye(2)
print(x)
print(x.ndim)
print(x.shape)
print(x.dtype)


x = np.array([1, 2, 3])
print(x)
print(x.reshape((1, 3)))
print(x.reshape((3, 1)))

x = np.arange(16)
print(x.reshape(4, 4))

x = np.arange(10)
print(x*2)


x = np.array([1, 2, 3])
y = np.array([1, 2, 3])
print(x+y)
print(x - y)
print(x*y)
print(x.dot(y))


x = np.arange(16)
y = x.reshape(4, 4)
print(y)
print(y[1:])
print(y[::-1])
print(y[::1])