
#    Copyright (C) 2009-2011 Jason T. Hopper
#    All rights reserved.                                                                                                                                                              
#  
#    Jason T. Hopper <hopperj@dal.ca>
#    Alice Research Group
#    1271 Church Street, Apt 713. Halifax, Nova Scotia, B3J 3L3  
#
#    ALICE is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    ALICE is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

#!/bin/python


import numpy as np
#from neural_network import NN
from newNN import NeuralNetwork as NN
from matplotlib import pylab



def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2


class NeuralNetwork:

    def __init__(self, layers, activation='tanh'):
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_prime = sigmoid_prime
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_prime = tanh_prime

        # Set weights
        self.weights = []
        # layers = [2,2,1]
        # range of weight values (-1,1)
        # input and hidden layers - random((2+1, 2+1)) : 3 x 3
        for i in range(1, len(layers) - 1):
            r = 2*np.random.random((layers[i-1] + 1, layers[i] + 1)) -1
            self.weights.append(r)
        # output layer - random((2+1, 1)) : 3 x 1
        r = 2*np.random.random( (layers[i] + 1, layers[i+1])) - 1
        self.weights.append(r)

    def fit(self, X, y, learning_rate=0.2, epochs=100000):
        # Add column of ones to X
        # This is to add the bias unit to the input layer
        ones = np.atleast_2d(np.ones(X.shape[0]))
        X = np.concatenate((ones.T, X), axis=1)
         
        for k in range(epochs):
            i = np.random.randint(X.shape[0])
            a = [X[i]]

            for l in range(len(self.weights)):
                    dot_value = np.dot(a[l], self.weights[l])
                    activation = self.activation(dot_value)
                    a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * self.activation_prime(a[-1])]

            # we need to begin at the second to last layer 
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1): 
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_prime(a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation 
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)

            if k % 10000 == 0: print 'epochs:', k

    def predict(self, x): 
        a = np.concatenate((np.ones(1).T, np.array(x)), axis=1)      
        for l in range(0, len(self.weights)):
            a = self.activation(np.dot(a, self.weights[l]))
        return a

if __name__ == '__main__':


    xor_states = [ [0,0], [1,0], [1,1], [0,1] ]
    xor_outputs = [ 0, 1, 0, 1 ]

    nn = NeuralNetwork([2,2,1])
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([0, 1, 1, 0])
    nn.fit(X, y)
    for e in X:
        print(e,nn.predict(e))













"""
nn = NN(2,3,1)
#nn.load()
nn.useThresholds = 1
nn.useSigmoid = 0
nn.max_itt = 10000
#print nn.getHOW()
#ihw = np.array([[-11.48012016, -5.97035322, -5.62449861],
# [  5.61543731, -6.20376074, 11.49647411]])

#how = np.array([[ 1.55603478],
# [-3.32338919],
# [-1.09690387]])

#nn.setIHW(ihw)
#nn.setHOW(ihw)
outputs, expected = nn.train_network( [ xor_states, xor_outputs ] )
nn.save()

print "Running test:"
print nn.run(xor_states[0])
nn.reset_nodes()
print nn.run(xor_states[1])
nn.reset_nodes()
print nn.run(xor_states[2])
nn.reset_nodes()
print nn.run(xor_states[3])
nn.reset_nodes()
print nn.getHOW()
"""



"""
pylab.plot( range(len(outputs)), outputs )
pylab.plot( range(len(expected)), expected )
pylab.show()
pylab.savefig("nn_training")
"""
