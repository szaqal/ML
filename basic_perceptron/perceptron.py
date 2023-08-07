#!/usr/bin/env python
import numpy as np

def compute_output(w, x):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i]
    if z <= 0:
        return -1
    else:
        return 1


def np_compute_output(w, x):
    # sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
    return np.sign(np.dot(w,x))


if __name__ == '__main__':
    print(compute_output([0.9, -0.6, -0.5], [1.0, -1.0, -1.0]))
    print(compute_output([0.9, -0.6, -0.5], [1.0, 1.0, 1.0]))
