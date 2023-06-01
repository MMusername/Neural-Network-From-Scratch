import random
import numpy as np
from error_functions import error, errorDerivative

class Layer():
    # creates layer with random weights and bias.
    def __init__(self, inputSize, outputSize, activation, derivative) -> None:
        self.__input = None
        self.__output = None
        self.__weights = np.random.rand(outputSize, inputSize)
        self.__bias = np.random.rand(outputSize, 1).T
        self.__activationFunction = activation
        self.__activationFunctionDerivative = derivative
     
    def getInputSize(self):
        return self.__weights.shape[1]

    def getOutputSize(self):
        return self.__weights.shape[0]

    def getWeights(self):
        return self.__weights

    def forwardPropagation(self, input):
        self.__input = input
        self.__output = self.__activationFunction(
            np.dot(self.__input, self.__weights.T) + self.__bias
        )
        return self.__output


    def backwardPropagation(self, outputGradient, learningRate):
        outputGradient = self.__activationFunctionDerivative(self.__output) * outputGradient
        
        # dE/dX = dE/dY * W (bez W.T?)
        grad = np.dot(outputGradient, self.__weights) 
        
        # dE/dW = X.T * dE/dY
        weightsGradient = np.dot(self.__input.T, outputGradient)
        self.__weights -= weightsGradient.T * learningRate

        # dE/dB = dE/dY
        self.__bias -= outputGradient * learningRate

        return grad


class Network:
    # creates network and checks size correctnes
    def __init__(self, sizes, functions, derivatives) -> None:
        sizesLength = len(sizes)
        if sizesLength < 2:
            raise ValueError("sizes list to small")
        if sizesLength - 1 != len(functions) or sizesLength - 1 != len(derivatives):
            raise ValueError("different list lengths")

        self.__network = []
        for i in range(sizesLength - 1):
            self.__network.append(Layer(sizes[i], sizes[i + 1], functions[i], derivatives[i]))
    
    
    def forward(self, input_data):
        output = input_data
        for layer in self.__network:
            output = layer.forwardPropagation(output)
        return output


    # dodać opcję nauki aż do zamierzonej dokładności
    # dodać % na którym ma się uczyć i % na którym testować
    # dodać sprawdzanie poprawności
    def learn(self, data, expected, iterations, learningRate):
        for i in range(iterations):
            e = 0
            for x, y in zip(data, expected):
                output = self.forward(x)

                e += error(y, output)
            
                # backward
                grad = errorDerivative(y, output)
                for layer in reversed(self.__network):
                    grad = layer.backwardPropagation(grad, learningRate)


            if i % 10 == 0:
                print(f'iteration: {i}, error: {e / data.size}')

    
    def chunkLearn(self, data, expected, chunkSize, iterations, learningRate):
        dataSize = data.shape[0]
        if chunkSize > dataSize:
            raise ValueError("chunkSize to big")
        
        for i in range(iterations):
            e = 0
            used = random.sample(range(0, dataSize), chunkSize)
            for j in range(chunkSize):
                x = data[used[j]]
                y = expected[used[j]]
                output = self.forward(x)
                e += error(y, output)
                grad = errorDerivative(y, output)
                for layer in reversed(self.__network):
                    grad = layer.backwardPropagation(grad, learningRate)

            if i % 10 == 0:
                print(f'iteration: {i}, error: {e / chunkSize}')

    def save(self, file):
        for layer in self.__network:
            print(layer.getInputSize())
        print(self.__network[-1].getOutputSize())
        
        for layer in self.__network:
            print(layer.getWeights())

