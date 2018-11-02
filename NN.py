# BASIC NEURAL NETWORK (NN)
# DESIGNED FROM MUHAMMAD USMAN TAHIR
# THE CODE INVOLVES THREE STEPS: BUILD, TRAIN, TEST

# First import libraries
import numpy as np
import matplotlib.pyplot as plt
from numpy import array, dot, exp, random

# Making a class of Neural Network
class Neural_Network():
    def __init__(self):

# Now generating some random weights as inputs to start of
# Using random number generator so it generates same number when executed
        random.seed(1)

# Modelling a single neuron with 3 inputs and 1 output
# Assiging a random weight to a 3x1 matrix with values in range of +1 to -1
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

# Defining the main function that is our artificial neural function (SIGMOID)
# Sigmoid is a function that describes a S-shaped curve
# We pass the sum of weighted inputs in to this function to normalize between 0 & 1
    def __sigmoid(self, z):
        return 1 / (1 + exp(-z))

# To update weights taking gradient of sigmoid function (Gradient Descent)
# It is the deravative of sigmoid function
    def __sigmoid_derivative(self, z):
        return z * (1 - z)

# Training the Neural Network using trial and error
# Weights will be adjusted each time   
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):

# Passing the training set through NN
            output = self.think(training_set_inputs)

# Calculating error
            error = training_set_outputs - output

# To adjust weights multiplying error with gradient of sigmoid
# Then multiply with input data
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

# Weight adjusting taking place
            self.synaptic_weights += adjustment

# NN working function
    def think(self, inputs):

# Data passing through NN (NN thinking) 
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == "__main__":

# Initializing a single NN
    neural_network = Neural_Network()

    print("Random weights to start with:")
    print(neural_network.synaptic_weights)
    
# Making a sample training set
# The training set consist of 5 examples each consisting of 3 input values
# 1 output value
    training_set_inputs = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0, 1]]).T
    print("Training Data is build")  

# Training NN using training set
# Training 50,000 times and making minor changes each time
    neural_network.train(training_set_inputs, training_set_outputs, 50000)

    print("New random weights after training:")
    print(neural_network.synaptic_weights)

# Finally testing the Neural Network
    print("Testing data as input [1, 1, 0])")
    print(neural_network.think(np.array([1, 1, 0])))
    print("CONGRATULATIONS YOU DESIGNED A NEURAL NETWORK")
        
