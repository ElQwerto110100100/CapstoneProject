#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tuesday Oct 2, 2018
@author: Madhuri Suthar, PhD Candidate in Electrical and Computer Engineering, UCLA
https://gist.github.com/jamesloyys/ff7a7bb1540384f709856f9cdcdee70d#file-neural_network_backprop-py

Last updated on 21/02/2020
Modified by Kyle Hogg to have multipul hidden layers and make chainging variables easier
"""
# Imports
import numpy as np 
      
# Each row is a training example, each column is a feature  [X1, X2, X3]
X=np.array(([0,0,1],[0,1,1],[1,0,1],[1,1,1]), dtype=float)
y=np.array(([0,1],[1,0],[1,0],[0,1]), dtype=float) #expected outputs

# Varibals for number of nodes
epocs = 20000
#array with each element being the number of nodes in each layer and the number of elements being the number of layers plus the output layer
hiddenLayers=np.array((1,2,y.shape[1]))

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
        for x in range(0,(hiddenLayers.size)-1):
            self.weights.append(np.random.rand(hiddenLayers[x],hiddenLayers[x+1]))
        self.output = np. zeros(hiddenLayers[2])
		
    def feedforward(self):
        self.layers = []
        self.layers.append(sigmoid(np.dot(self.input, self.weights[0])))
        for i in range(0,(hiddenLayers.size)-1):
            self.layers.append(sigmoid(np.dot(self.layers[i], self.weights[i+1])))
        return self.layers[(hiddenLayers.size)-1]
        
    def backprop(self):
    #output layer weights
        d_weightsa = 2*(self.y -self.output)*sigmoid_derivative(self.output)
        d_weights3 = np.dot(self.layers[1].T, d_weightsa)
    #2nd Hidden node layer weights    
        d_weightsa = np.dot(d_weightsa, self.weights[2].T)*sigmoid_derivative(self.layers[1])
        d_weights2 = np.dot(self.layers[0].T, d_weightsa)
    #1st Hidden node layer weights
        d_weightsa = np.dot(d_weightsa, self.weights[1].T)*sigmoid_derivative(self.layers[0])
        d_weights1 = np.dot(self.input.T, d_weightsa)
    
    #saving the weights)
        self.weights[0] += d_weights1
        self.weights[1] += d_weights2
        self.weights[2] += d_weights3

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()
        

NN = NeuralNetwork(X,y)
for i in range(epocs): # trains the NN x times
    
    if i % (2000) ==0: 
        print ("for iteration # " + str(i) + "\n")
        print ("Input : \n" + str(X))
        print ("Actual Output: \n" + str(y))
        print ("Predicted Output: \n" + str(NN.feedforward()))
        print ("Loss: \n" + str(np.mean(np.square(y - NN.feedforward())))) # mean sum squared loss
        print ("\n")
  
    NN.train(X, y)
