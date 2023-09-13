"""
The idea of neural networks is to take the linear model and add more layers of neurons
to it. The first layer is called the input layer, the last layer is called the output layer,
and the layers in between are called hidden layers. Each layer has a set of weights and
biases associated with it. The weights and biases are learned during the training process.
The more layers a neural network has, the more complex functions it can learn.

This is a simple example of how pyTorch creates a fully connected neural network in
an object-oriented fashion.
"""

import torch
import torch.nn as nn

# Define the class
class Net(nn.Module):
    """ Class inherited from nn.Module """
    def __init__(self):
        super(Net, self).__init__()  # Call the constructor of the parent class

        # (weights and biases are initialized automatically)
        self.fc1 = nn.Linear(10,20)     # 10 inputs, 20 outputs
        self.fc2 = nn.Linear(20,20)     # 20 inputs, 20 outputs
        self.output = nn.Linear(20,4)   # 20 inputs, 4 outputs

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.output(x)
        return x


# Instantiate the class
input_layer = torch.rand(10)
net = Net()
result = net(input_layer)
