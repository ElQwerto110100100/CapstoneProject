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
import csv

def defineOutput(outputNames):
    numOfOutputs = len(set(outputNames))
    outputNames = set(outputNames)
    outputArray = [None]*numOfOutputs
    for index, names in enumerate(outputNames):
        outputArray[index] = ([0]*numOfOutputs)
    for index, output in enumerate(outputNames):
        for jndex, zero in enumerate(outputArray):
            outputArray[index][jndex] = (1 if (index+1)/(jndex+1) == 1 else 0)
    return outputArray

    #highest number (greater than 0) is use to divide the dataset
    #if there are negative numbers add everything but the largest negative numbers
    #change all output to number array of 1, 0
def NormilizeData(data):
    maxNum = 0.0
    minNum = 0.0

    #find largest negative number
    for listItem in data:
        for item in listItem:
            try:
                compar = float(item)
                if compar < minNum:
                    minNum = compar
            except:
                #if it checks th id, pass
                pass

    # add every number by that negative
    if minNum < 0:
        for index, listItem in enumerate(data):
            for jndex, item in enumerate(listItem):
                try:
                    compar = float(item)
                    data[index][jndex] = compar - minNum
                except:
                    pass

#find largest positive number
    for listItem in data:
        for item in listItem:
            try:
                compar = float(item)
                if compar > maxNum:
                    maxNum = compar
            except:
                #if it checks th id, pass
                pass

    #normilize dataset
    for index, listItem in enumerate(data):
        for jndex, item in enumerate(listItem):
            try:
                compar = float(item)
                data[index][jndex] = compar / maxNum
            except:
                pass

    for i in data:
        print(i)
    #read the csv file
def csvReader(fname):
    with open(fname, newline='') as csvfile:
        datasetReader = csv.reader(csvfile, delimiter=',', quotechar='|')
        data = [row for row in datasetReader]
    #strip data
    output = [data[index][4] for index, name in enumerate(data)]
    output = output[1:] # get ride of top labels
    output = [name for name in set(output)]#remove duplicates
    outputId = defineOutput(output)#set every output to a id
    data = data[1:]#remove label row
    # replace output with a output Id
    for index, name in enumerate(data):
        for jndex, id in enumerate(outputId):
            data[index][4] = outputId[jndex] if data[index][4] == output[jndex] else  data[index][4]

    NormilizeData(data)

csvReader("IRIS.csv")

"""
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
"""
