"""
Activation functions are used to introduce non-linearity in the neural networks.
They are inserted between the layers of the neural network to deal high higHly
non-linear datasets. They are applied to the output of the neurons and their purpose is to decide whether
the neuron should be activated or not.

The most common activation function is ReLU (Rectified Linear Unit), which
is defined as:

ReLU(x) = max(0, x)
"""

import torch
import torch.nn as nn

# Create a tensor
x = torch.tensor([0.0401, -0.9005, 0.0397, -0.0876])

# Create random weights
weight_1 = torch.tensor([[-0.1094, -0.8285,  0.0416, -1.1222],
                        [0.3327, -0.0461,  1.4473, -0.8070],
                        [0.0681, -0.7058, -1.8017,  0.5857],
                        [0.8764,  0.9618, -0.4505,  0.2888]])

weight_2 = torch.tensor([[ 0.6856, -1.7650,  1.6375, -1.5759],
                        [-0.1092, -0.1620,  0.1951, -0.1169],
                        [-0.5120,  1.1997,  0.8483, -0.2476],
                        [-0.3369,  0.5617, -0.6658,  0.2221]])

weight_3 = torch.tensor([[ 0.8824,  0.1268,  1.1951,  1.3061],
                        [-0.8753, -0.3277, -0.1454, -0.0167],
                        [ 0.3582,  0.3254, -1.8509, -1.4205],
                        [ 0.3786,  0.5999, -0.5665, -0.3975]])

# Instantiate the ReLU activation function
relu = nn.ReLU()

# Apply the ReLU to the hidden layers
hidden_1_activated = relu(torch.matmul(x, weight_1))
hidden_2_activated = relu(torch.matmul(hidden_1_activated, weight_2))
y = torch.matmul(hidden_2_activated, weight_3)

print(y)

# Apply the ReLU to the combined weight
weight_composed_1_activated = relu(torch.matmul(weight_1, weight_2))
weight_composed_2_activated = relu(torch.matmul(weight_composed_1_activated, weight_3))
y2 = torch.matmul(x, weight_composed_2_activated)

print(y2)





