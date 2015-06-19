
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
import cPickle

class NN:


    def __init__(self, inputs=2, hidden=3, outputs=1, N=1.0e-1, max_itt=1e3):
        # Set default values
        self.threshold = 0.9
        self.num_inputs = inputs+1
        self.num_hidden = hidden+1
        self.num_outputs = outputs
        self.N = N
        self.max_itt = max_itt
        self.useThresholds = 0
        self.useSigmoid = 0
        
        self.inputs = np.zeros((self.num_inputs,))
        self.hidden = np.zeros((self.num_hidden,))
        self.outputs = np.zeros((self.num_outputs,))
        
        # can randomize the initial weights to anything but from what I've
        # tried there really needs to be some negatives in there and its best
        # not not have any too close to 1.
        self.randomIHW()
        self.randomHOW()
        # d_ihw and d_how will be the summed changes in weights to be applied
        # after looping through each element in the training set. 
        # See theory for why it is done like this.
        self.d_ihw = np.zeros((self.num_inputs, self.num_hidden))
        self.d_how = np.zeros((self.num_hidden, self.num_outputs))

    def randomIHW(self):
        # Random numbers [-1.0, 1.0]
        self.ihw = 2.0*np.random.randn(self.num_inputs, self.num_hidden)-1.0

    def randomHOW(self):
        # Random numbers [-1.0, 1.0]
        self.how = 2.0*np.random.randn(self.num_hidden, self.num_outputs)-1.0
        
    def setIHW(self, weights):
        if weights.shape != self.ihw.shape or type(weights[0]) != float:
            # number of elements is wrong. Signal there was an error
            return 1
        self.ihw = weights
        
    def setHOW(self, weights):
        if weights.shape != self.how.shape and type(weights[0]) != float:
            # number of elements is wrong. Signal there was an error
            return 1
        self.how = weights

    def getIHW(self):
        return self.ihw

    def getHOW(self):
        return self.how
    
    def save(self):
        # Store ihw and how into a pickle
        f = open('nn.pickle', 'w')
        cPickle.dump( [self.ihw, self.how], f)
        f.close()
        
    def load(self):
        # Load ihw and how from pickle
        self.ihw, self.how = cPickle.load(open('nn.pickle'))
        # Make sure we have the right values for node numbers.
        self.num_inputs, self.num_hidden = self.ihw.shape
        self.num_hidden, self.num_outputs = self.how.shape
        
        
    def sigmoid(self,x):
        return 1.0/( 1.0 + np.e**(-x) )
    
    def dsigmoid(self,x):
        return x*( 1.0 - x )
        
        
    def reset_nodes(self):
        # Clear all the energies
        self.inputs = np.zeros((self.num_inputs,))
        self.outputs = np.zeros((self.num_outputs,))
        self.hidden = np.zeros((self.num_hidden,))
               
    def run(self, inpt ):
        self.reset_nodes()
        self.inputs[:len(inpt)] = np.array( inpt )
        #print self.inputs.shape
        self.inputs[self.num_inputs-1] = 1.0
        #print "Running:",self.input
        ret = self.feedforward()
        #print "Got:",ret
        return ret
        
    def feedforward(self, ins=None):
        if ins is not None:
            self.inputs = np.array( ins )
        # Make an array to be used at the end
        ans = np.zeros(( self.num_outputs,))
        # Sum energies going into each hidden node
        for h in np.arange( self.num_hidden ):
            self.hidden[h] = np.sum( self.inputs*self.ihw[:,h] )

        if self.useSigmoid:
            # Sigmoid all the hidden node energies
            self.hidden = self.sigmoid( self.hidden )
        
        # Sum all the energies going into output nodes
        for o in np.arange( self.num_outputs ):
            self.outputs[o] = np.sum( self.hidden*self.how[:,o] )

        if self.useSigmoid:
            # Sigmoid all the output node energies
            self.outputs = self.sigmoid( self.outputs )
            
        # Use threshold value to return an array of boolean values
        # meaning activated or not.
        if self.useThresholds:
            for o in np.arange( self.num_outputs ):
                if self.outputs[o] > self.threshold:
                    ans[o] = 1.0
            return ans
        return self.outputs
        
        
    def backpropagate(self, targets=np.array([])):
        
        # Array of output node errors
        o_err = self.dsigmoid( self.outputs )*( targets - self.outputs )        
        # Array for hidden node energy errors
        h_err = np.zeros((self.num_hidden,))
        
        # Calculate all the hidden node errors and sum the hidden-->output
        # weight changes to be applied later
        for h in np.arange( self.num_hidden ):
            self.d_how[h,:] += o_err*self.hidden[h] # self.N*
            h_err[h] = self.dsigmoid(self.hidden[h])*np.sum(o_err*self.how[h,:])
        
        # Calculate and sum all the input-->hidden weights and apply later
        for i in np.arange( self.num_inputs ):
            self.d_ihw[i,:] += h_err*self.inputs[i] # self.N*
            
        # Return the sum of the absolute value of output node errors
        #return np.sum( np.abs( o_err ) )
        return np.sum( o_err )*self.N
            
    def update_weights(self):
        # Update all weights at once
        self.ihw += self.d_ihw
        self.how += self.d_how
        # Clear the summs
        self.d_ihw = np.zeros((self.num_inputs, self.num_hidden))
        self.d_how = np.zeros((self.num_hidden,self.num_outputs))
       
       
    def train_network(self, X=None, Y=None):
        # should be a list of list pairs [ [inputs,outputs] ]
        i=0
        err = 0.0
        l_err = 0.0
        # An array of indecies where we should print out the error status
        special = np.arange(1, self.max_itt, float(self.max_itt)/50.0 )
        # Begin training loop
        errors = []
        while i < self.max_itt:
            # reset current error
            err = 0.0
            for ins, outs in zip(X,Y):
                
                self.run( np.array( ins ) )
                # Carry out backpropagation and keep track of the error sum
                err += abs( self.backpropagate( np.array(outs) ) )
            if i in special:
                # Print out status
                print "\033[1FError:",err," Is decreasing?",err < l_err,'       '
            # Do all weight updates at once after a full training cycle
            self.update_weights()
            # Remember this error before it is reset so we know if the error
            # is decreasing
            l_err = err
            # If the error is flat at 0, we are done!
            errors.append(err)
            #if err == 0.0 and l_err == 0.0:
            #    break
            # increase counter
            i+=1
            # Check to see if the error has dropped below some point. If 
            # it gets really low then we don't need to keep going. Over
            # training the system can lead to loss of generalization
            #if err < 1.0e-2:
            #    print "Error is below threshold now"
            #    print "Took",i," itterations. (",100.0*float(i)/self.max_itt,'%)'
            #    break
        print "Finished",i,'training loops.'
        print "Error:",err
        return errors





if __name__ == '__main__':
    import numpy as np
    from matplotlib import pylab as pl

    xor_states = np.array( [ [0,0], [0,1], [1,1], [1,0] ] )
    xor_outputs = np.array( [ 0, 1, 0, 1 ] )
    
    nn = NN(2,3,1)
    #nn.load()
    nn.useThresholds = 0
    nn.useSigmoid = 1
    nn.max_itt = 10000
    nn.N = 2.0e-10
    errors = nn.train_network( X = xor_states, Y = xor_outputs )
    print "Training complete"
    nn.save()
    nn.useThresholds=1
    print "Running test:"
    wrong = 0
    right = 0
    for i in range(xor_outputs.shape[0]):
        res = nn.run(xor_states[i])
        print res
        print "Expected:",xor_outputs[i]
        print "\n"
        if res[0] == xor_outputs[i]:
            right+=1
        else:
            wrong+=1

    print "%d of %d correct"%(right,wrong+right)
    pl.plot( range(len(errors)), errors)
    pl.show()
