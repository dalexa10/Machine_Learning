"""
Neural networks (another name for deep learning models) are a framework for
learning from data. They are used for a variety of tasks, including image
classification, speech recognition, and natural language processing.
"""

import torch
import numpy as np

# Create a tensor (a multi-dimensional array)
x = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(x)

# Create a tensor with random values
x = torch.rand(3, 3)
print(x)

# Multiple two tensors
y = torch.rand(3, 3)
z = torch.mm(x, y)  # Same as torch.matmul(x, y)
print(z)

# Element-wise multiplication
z = x * y
print(z)

# Special matrices
x = torch.ones(3, 3)    # Ones tensor
print(x)

x = torch.zeros(3, 3)   # Zeros tensor
print(x)

x = torch.eye(3, 3)     # Identity tensor
print(x)

# Create a tensor from a numpy array
x_numpy = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
x = torch.from_numpy(x_numpy)
print(x)
