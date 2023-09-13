"""
Loss Functions:
- Initialize neural networks with random weights and biases
- Do a forward pass
- Calculate the loss function (1 number)
- Calculate the gradients of the loss function with respect to the weights and biases
- Update the weights and biases

Now, what is a loss function? A loss function is a function that measures how well
the neural network is doing.

For regression problems, the most common loss function is the mean squared error (MSE), which is defined as
MSE = 1/N * sum((y_true - y_pred)^2)

For classification: softmax cross entropy is the most common used one, which is a proxy loss function
that is differentiable (instead of accuracy, which is not) and is defined as:

For more complicated problems (object dectection) there are more complicated loss functions (see later)

_______________________________________________________________________________________________________________
Recall, one of the most important feature of loss functions is that they are differentiable. This is because
the gradients of the loss function are used to update the weights and biases of the neural network.
_______________________________________________________________________________________________________________

"""

# Example

import torch
import torch.nn as nn

# Initialize the score and ground truth
logits = torch.tensor([[-1.2, 0.12, 4.8]])
ground_truth = torch.tensor([2])

# Instantiate the cross entropy loss function
cross_entropy = nn.CrossEntropyLoss()

# Calculate the loss
loss = cross_entropy(logits, ground_truth)
print(loss)
