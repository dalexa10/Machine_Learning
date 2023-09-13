"""
This is a simple example of how pyTorch computes the gradients of a function
Internally, pyTorch uses a computational graph to keep track of the gradients
and computes the gradients using the chain rule (automatic differentiation).
"""

import torch

# Define tensors
x = torch.tensor(-3., requires_grad=True)
y = torch.tensor(5., requires_grad=True)
z = torch.tensor(-2., requires_grad=True)

# Functions
q = x + y
f = q * z

# Compute gradients
f.backward()

print("Gradient of z is {:.3f}".format(z.grad))
print("Gradient of y is {:.3f}".format(y.grad))
print("Gradient of x is {:.3f}".format(x.grad))
