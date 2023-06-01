import numpy as np

def tanh(x):
    return np.tanh(x)

def tanhDerivative(x):
    return 1 - np.tanh(x) ** 2 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoidDerivative(x):
    y = sigmoid(x)
    return y * (1 - y)

def ReLU(x):
    for i in range(np.prod(x[0].shape)):
        x[0][i] = max(0, x[0][i])
    return x

def ReLUDerivative(x):
    return x > 1