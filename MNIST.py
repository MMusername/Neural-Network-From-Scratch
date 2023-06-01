from neural_network import Network
import numpy as np
from activation_functions import *
import os

from keras.datasets import mnist

from matplotlib import pyplot

def predictedValue(prediction):
    currentMax = -10
    j = 0
    for i in range(prediction[0].size):
        if prediction[0][i] > currentMax:
            currentMax = prediction[0][i]
            j = i
    return j


(train_X, train_y), (test_X, test_y) = mnist.load_data()

os.system("clear")

'''
for i in range(9):  
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()
'''

# zmiana X
Xtrain = np.reshape(train_X, (60000, 1, 28 * 28))
Xtest = np.reshape(test_X, (10000, 1, 28 * 28))

# zmiana Y
Ytrain = []
for y in train_y:
    z = np.zeros(10)
    z[y] = 1
    Ytrain.append(z)
Ytrain = np.array(Ytrain)

network = Network((28 * 28, 60, 40, 20, 10), 
    (tanh, tanh, tanh, tanh), 
    (tanhDerivative, tanhDerivative, tanhDerivative, tanhDerivative))
network.chunkLearn(data=Xtrain, expected=Ytrain, chunkSize=100, iterations=1000, learningRate=0.1)

# test
print("testing started")
correct = 0
#i = 0 # a jebać ładny styl
for x, y in zip(Xtest, test_y):
    predicted = predictedValue(network.forward(x))
    if predicted == y:
        correct += 1

    '''
    print(predicted)
    pyplot.imshow(test_X[i], cmap=pyplot.get_cmap('gray'))
    pyplot.show()
    i += 1
    '''

print(f'OVERALL SCORE: ', correct / test_y.size)
