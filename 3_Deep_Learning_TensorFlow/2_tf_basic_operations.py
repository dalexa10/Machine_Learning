"""
Show some examples of basic operations in tensorflow
Execution in Pycharm seems to crash for some reason when running in Python console
#TODO: Debug this issue
Suggestion: run in terminal with a conda environment (or with iPython)
"""

from tensorflow import constant, add
import tensorflow as tf

# Define 0-dimensional tensors
a0 = constant([1])
a1 = constant([2])

# Define 1-dimensional tensors
b0 = constant([1, 2])
b1 = constant([3, 4])

# Define 2-dimensional tensors
c0 = constant([[1, 2], [3, 4]])
c1 = constant([[5, 6], [7, 8]])

# Addition
# The add operation performs element-wise addition with two tensors.
a2 = add(a0, a1)  # Equivalent to a0 + a1
b1 = add(b0, b1)
c2 = add(c0, c1)

# Print the tensors
print(a2.numpy())
print(b1.numpy())
print(c2.numpy())

# Element-wise multiplication
# The multiply operation performs element-wise multiplication with two tensors.
a3 = a0 * a1  # Equivalent to tf.multiply(a0, a1)
b3 = b0 * b1
c3 = c0 * c1

# Print the tensors
print(a3.numpy())
print(b3.numpy())
print(c3.numpy())

# Matrix multiplication
# The matmul operation performs matrix multiplication with two tensors.
d0 = tf.constant([[1, 2], [3, 4], [5, 6]])
d1 = tf.constant([[7, 8, 9], [10, 11, 12]])
d2 = tf.matmul(d0, d1)

# Print the tensors
print(d2.numpy())

