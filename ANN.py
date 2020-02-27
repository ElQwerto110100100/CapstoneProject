#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Oct 2, 2018
@author: Madhuri Suthar, PhD Candidate in Electrical and Computer Engineering, UCLA
https://gist.github.com/jamesloyys/ff7a7bb1540384f709856f9cdcdee70d#file-neural_network_backprop-py

Last updated on 21/02/2020

Change log
Kyle Hogg - 26/02/2020 - Make number of hidden nodes and layers as a variable.

"""
# Imports
import numpy as np 
#import clpy as cp 
      
# Each row is a training example, each column is a feature  [X1, X2, X3]
X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0,1],[1,0],[1,0],[0,1]), dtype=float) #expected outputs

# Varibals for number of nodes
epocs = 20000
#array with each element being the number of nodes in each layer and the number of elements being the number of layers plus the output layer
hiddenLayers=np.array((3,2,y.shape[1]))

# Define useful functions    

# Activation function
def sigmoid(t):
    return 1/(1+np.exp(-t))

# Derivative of sigmoid
def sigmoid_derivative(p):
    return p * (1 - p)

# Class definition
class NeuralNetwork:
    def __init__(self, x,y):
        self.input = x
        self.y = y
        self.weights = []
        self.weights.append(np.random.rand(self.input.shape[1],hiddenLayers[0])) # considering we have 4 nodes in the hidden layer
        for i in range(0,(hiddenLayers.size)-1):
            self.weights.append(np.random.rand(hiddenLayers[i],hiddenLayers[i+1]))
            j=i
        self.output = np. zeros(hiddenLayers[j+1])
		
    def feedforward(self):
        self.layers = []
        self.layers.append(sigmoid(np.dot(self.input, self.weights[0])))
        for i in range(0,(hiddenLayers.size)-1):
            self.layers.append(sigmoid(np.dot(self.layers[i], self.weights[i+1])))
        return self.layers[(hiddenLayers.size)-1]
	
    def backprop(self):
        
    #output layer weights	
        d_weightsb = []
        d_weightsa = 2*(self.y -self.output)*sigmoid_derivative(self.output)
        for i in range((hiddenLayers.size -1), 0, -1):
            d_weightsb.append(np.dot(self.layers[i-1].T, d_weightsa))
            d_weightsa = np.dot(d_weightsa, self.weights[i].T)*sigmoid_derivative(self.layers[i-1])
        d_weightsb.append(np.dot(self.input.T, d_weightsa))
        
    # #saving the weights)
        for i in range(0, len(self.weights)):
            self.weights[i] += d_weightsb[((len(self.weights)-1)-i)]

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
        

NN = NeuralNetwork(X,y)
for i in range(epocs): # trains the NN x times
    
    if i % (epocs/10) ==0: 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
        print ("\n")
  
    NN.train(X, y)
