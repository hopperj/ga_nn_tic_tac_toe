#!/bin/python


import numpy as np
from neural_network import NN






xor_states = np.array( [ [0,0], [0,1], [1,1], [1,0] ] )
xor_outputs = np.array( [ 0, 1, 0, 1 ] )


net = NN()

net.load()
net.useThresholds = 1
net.useSigmoid = 1


for inpt, out in zip( xor_states, xor_outputs ):
    print net.run( inpt )
    print out
    print "\n\n"    
