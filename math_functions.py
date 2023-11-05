import numpy as np


def pi(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        if -0.5 < x[i] < 0.5:
            y[i] = 1
        elif x[i] == -0.5 or x[i] == 0.5:
            y[i] = 0.5
    return y


def tri(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        if abs(x[i]) < 1:
            y[i] = 1 - abs(x[i])
    return y


def sinc(x):
    n = len(x)
    y = np.zeros(n)
    sin = np.sin(x)
    for i in range(n):
        if x[i] != 0:
            y[i] = sin[i]/x[i]
        else:
            y[i] = 1
    return y
