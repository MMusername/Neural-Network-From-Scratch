from neural_network import Network
import numpy as np
from activation_functions import *

def main():
    network = Network([2, 3, 1], [tanh, tanh], [tanhDerivative, tanhDerivative])

    X = np.array([[[0, 0]], [[0, 1]], [[1, 0]], [[1, 1]]])
    Y = np.array([[0], [1], [1], [0]])

    #network.learn(X, Y, 1500, 0.1)
    network.chunkLearn(X, Y, 3, 1500, 0.1)

    for x, y in zip(X, Y):
        print(f"predicted: {network.forward(x)}, expected: {y}")
    

    network.save(None)
    

if __name__ == "__main__":
    main()
