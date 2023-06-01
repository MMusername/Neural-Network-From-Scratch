import numpy as np

def error(correctOutput, output):
    return np.mean(np.power(correctOutput - output, 2))

def errorDerivative(correctOutput, output):
    return 2 * (output - correctOutput) / np.size(correctOutput)