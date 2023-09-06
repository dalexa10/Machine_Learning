"""
Brief ideas:
- Tensorflow is a library for numerical computation using data flow graphs.
- Tensors are a generalization of vectors and matrices to potentially higher dimensions.
- Something that I'll always remember is what a Professor Eric Shaffer from UIUC said "you can think of tensor as
a machine that takes some vectors and spits out some other vectors in a linear fashion."

"""
# These are just simple examples of how to declare tensors and perform operations on them.

import tensorflow as tf

# Declaring tensors
d0 = tf.ones((1,))  # Scalar
d1 = tf.ones((2,))  # Vector
d2 = tf.ones((2, 2))  # Matrix
d3 = tf.ones((2, 2, 2))  # 3-Tensor

# Printing tensors
print(d0.numpy())
print(d1.numpy())
print(d2.numpy())
print(d3.numpy())

# Define a constant (Recall, constants are immutable)
a = tf.constant(3, shape=[2, 3])  # 2x3 matrix of 3s
print(a.numpy())

# Create a zero filled tensor from a given shape from other tensor
b = tf.zeros_like(a)
print(b.numpy())

# Define a variable (Recall, variables are mutable)
c = tf.Variable([1, 2, 3, 4, 5, 6], dtype=tf.float32)  # Vector of 6 elements
print(c.numpy())

# Multiplying tensors (This is a dot product)
d = tf.multiply(a, b)
print(d.numpy())

# Convert back to numpy array
e = d.numpy()
print(e)




