# Basic Neural Network:
Here a neural network is designed to showcase the working methodology of a simple neural network and coding basic neural network.

This simple project displays the working of a basic neural network and the mathematics that is involved in developing a neural nets, so lets get started:

First we will start by discussing the concept of Machine Learning because neural nets are the core feature of Deep Learning that is a sub field of Machine Learning.

So Whats is Machine Learning?

# Machine Learning:
Machine Learning is a branch of Science, Engineering and Mathematics that is involved in forming algorithms to solve complex mathematical and computing problems such as image recognition, Language Understanding, object classification and many other real world problems by identifiying patterns and rules from training data set based on the provided features, attributes or characteristics. Then with the help of these Machine generated rules the computer or Processing system process testing data and give correct results with great accuracy.

Now lets discuss Deep Learning and then we will code a simple Neural Network:

# Deep Learning:
Deep learning is the sub brach of Machine Learning, it is actually based on the concept of designing such Machine Learning algorithms using mathematical functions and logics that displays the comlex working priciple of human brains that is based on millions or billions of neurons, so in Deep Learning these mathematical designed functions are called as Artificial neural networks or simply neural nets. These neural nets can be of multiple layers, when they have multile hidden layers then they are termed as Convolution Neural Networks and come under the hood of Deep Learning. These are very powerfull tools and can be used for solving vast amount of complex problems efficiently.

Ok, so lets get started by designing our own neyral network using python and other supportive libraries.

first importing numpy (A pyhton library for performing matrix operations i.e arrays and other mathematic tools), then matplotlib.pyplot is matlab based library used for plotting results, although in this neural network we did'nt plot anyhting as it was not necessary, but its a good habit to import these two basic libraries always while coding. Other libraries required for ML are pandas, tensorflow, scikit learn and some others as well. They are used for designing more complex neural nets and ML algorithms.  

import numpy as np
import matplotlib.pyplot as plt
from numpy import array, dot, exp, random

Making a class of Neural Network

class Neural_Network():
    def __init__(self):

Now generating some random weights as inputs to start of, Using random number generator so it generates same number when executed

        random.seed(1)

Modelling a single neuron with 3 inputs and 1 output,Assiging a random weight to a 3x1 matrix with values in range of +1 to -1

        self.synaptic_weights = 2 * random.random((3, 1)) - 1

Defining the main function that is our artificial neural function (SIGMOID), Sigmoid is a function that describes a S-shaped curve and represents very closely to neurons in brain. We pass the sum of weighted inputs in to this function to normalize between 0 & 1

    def __sigmoid(self, z):
        return 1 / (1 + exp(-z))

To update weights taking gradient of sigmoid function (Gradient Descent), It is the deravative of sigmoid function

    def __sigmoid_derivative(self, z):
        return z * (1 - z)

Training the Neural Network using trial and error, Weights will be adjusted each time  

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):

Passing the training set through NN

            output = self.think(training_set_inputs)

Now calculating error to train neural network with so that weights are adjusted to most accurate value

            error = training_set_outputs - output

To adjust weights multiplying error with gradient of sigmoid, Then multiply with input data

            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))

Weight adjusting taking place

            self.synaptic_weights += adjustment

NN working function

    def think(self, inputs):

Data passing through NN (NN thinking) 

        return self.__sigmoid(dot(inputs, self.synaptic_weights))

Making main function for processing training data and generating results

if __name__ == "__main__":

Initializing a single NN

    neural_network = Neural_Network()

    print("Random weights to start with:")
    print(neural_network.synaptic_weights)
    
Making a sample training set, The training set consist of 5 examples each consisting of 3 input values & 1 output value

    training_set_inputs = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0, 1]]).T
    print("Training Data is build")  

Training NN using training set, Training 50,000 times and making minor changes each time

    neural_network.train(training_set_inputs, training_set_outputs, 50000)

    print("New random weights after training:")
    print(neural_network.synaptic_weights)

Finally testing the Neural Network

    print("Testing data as input [1, 1, 0])")
    print(neural_network.think(np.array([1, 1, 0])))
    print("CONGRATULATIONS YOU DESIGNED A NEURAL NETWORK")
    
Cheers, you have designed your own neural network and I believe it was quite easy, with this basic knowledge more advanced neural networks can be designed to solve really complex problems.
        






