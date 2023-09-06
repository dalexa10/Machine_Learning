import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


def get_gradients(variable):
    """ Compute the first derivatives of `variable`
    with respect to child variables.
    """
    gradients = defaultdict(lambda: 0)

    idx = 0
    def compute_gradients(variable, path_value):
        nonlocal idx
        for child_variable, local_gradient in variable.local_gradients:
            # "Multiply the edges of a path":
            value_of_path_to_child = path_value * local_gradient

            # "Add together the different paths":
            gradients[child_variable] += value_of_path_to_child

            # recurse through graph:
            compute_gradients(child_variable, value_of_path_to_child)

            idx += 1


    compute_gradients(variable, path_value=1)
    print('Derivatives taken: {}'.format(idx))
    # (path_value=1 is from `variable` differentiated w.r.t. itself)
    return gradients


class Variable:
    def __init__(self, value, local_gradients=list()):
        self.value = value
        self.local_gradients = local_gradients

    def __add__(self, other):
        return add(self, other)

    def __mul__(self, other):
        return mul(self, other)

    def __sub__(self, other):
        return add(self, neg(other))

    def __truediv__(self, other):
        return mul(self, inv(other))
    def __sin__(self):
        return sin(self)

    def __exp__(self):
        return exp(self)

    def __log__(self):
        return log(self)
def add(a, b):
    value = a.value + b.value
    local_gradients = (
        (a, 1),
        (b, 1)
    )
    return Variable(value, local_gradients)


def mul(a, b):
    value = a.value * b.value
    local_gradients = (
        (a, b.value),
        (b, a.value)
    )
    return Variable(value, local_gradients)


def neg(a):
    value = -1 * a.value
    local_gradients = (
        (a, - 1),
    )
    return Variable(value, local_gradients)


def inv(a):
    value = 1 / a.value
    local_gradients = (
        (a, -1 / (a.value ** 2)),
    )
    return Variable(value, local_gradients)

def sin(a):
    value = np.sin(a.value)
    local_gradients = (
        (a, np.cos(a.value)),
    )
    return Variable(value, local_gradients)

def exp(a):
    value = np.exp(a.value)
    local_gradients = (
        (a, np.exp(a.value)),
    )
    return Variable(value, local_gradients)

def log(a):
    value = np.log(a.value)
    local_gradients = (
        (a, 1/a.value),
    )
    return Variable(value, local_gradients)


if __name__ == '__main__':
    import sympy as sym


    # -----------------------------------------------------------------------
    #                          FUNCTION ONE
    # -----------------------------------------------------------------------
    a = Variable(230.3)
    b = Variable(33.2)

    def f(a, b):
        return (a / b - a) * (b / a + a + b) * (a - b)

    y = f(a, b)

    gradients = get_gradients(y)
    print('---------------------------------------------------\n'
            'Automatic differentiation results:')
    print("The partial derivative of y with respect to a =", gradients[a])
    print("The partial derivative of y with respect to b =", gradients[b])
    print('---------------------------------------------------\n')

    # -----------------------------------------------------------------------
    #           Veryfication with other derivatives methods
    # -----------------------------------------------------------------------
    #                       Symbolic differentiation
    # -----------------------------------------------------------------------
    x, y = sym.symbols('x y')
    a_num, b_num = 230.3, 33.2

    f_sym = (x / y - x) * (y / x + x + y) * (x - y)
    df_dx = sym.diff(f_sym, x)
    df_dy = sym.diff(f_sym, y)
    print('------------------------------------ \n'
          'Symbolic differentiation results:')
    print('Derivative of f(x, y) w.r.t x is: ', df_dx.subs([(x, a_num), (y, b_num)]))
    print('Derivative of f(x, y) w.r.t y is: ', df_dy.subs([(x, a_num), (y, b_num)]))
    print('------------------------------------\n')


    def f(x, y):
        """
        Function definition that will be used to test different differentiation methods
        """
        return (x / y - x) * (y / x + x + y) * (x - y)


    # -----------------------------------------------------------------------
    #             Numerical differentiation (Finite difference)
    # -----------------------------------------------------------------------
    h = 1e-6
    df_dx = (f(a_num + h, b_num) - f(a_num, b_num)) / h
    df_dy = (f(a_num, b_num + h) - f(a_num, b_num)) / h

    print('-------------------------------------------- \n'
          'Numerical differentiation (Finite difference):')
    print('Derivative of f(x, y) w.r.t x is: ', df_dx)
    print('Derivative of f(x, y) w.r.t y is: ', df_dy)
    print('--------------------------------------------\n')

    # -----------------------------------------------------------------------
    #             Numerical differentiation (Complex step)
    # -----------------------------------------------------------------------
    h = 1e-6
    df_dx = (f(a_num + h * 1j, b_num).imag) / h
    df_dy = (f(a_num, b_num + h * 1j).imag) / h

    print('-------------------------------------------- \n'
            'Numerical differentiation (Complex step):')
    print('Derivative of f(x, y) w.r.t x is: ', df_dx)
    print('Derivative of f(x, y) w.r.t y is: ', df_dy)
    print('--------------------------------------------\n')


    # -----------------------------------------------------------------------
    #                          FUNCTION TWO
    # -----------------------------------------------------------------------
    a = Variable(2)
    b = Variable(3)
    c = Variable(6)

    def f(a, b, c):
        return (sin(a) * log(b)) + (exp(b) * c) + (a * c)

    y = f(a, b, c)

    gradients = get_gradients(y)
    print('---------------------------------------------------\n'
            'Automatic differentiation results:')
    print("The partial derivative of y with respect to a =", gradients[a])
    print("The partial derivative of y with respect to b =", gradients[b])
    print("The partial derivative of y with respect to c =", gradients[c])
    print('---------------------------------------------------\n')

    # -----------------------------------------------------------------------
    #           Veryfication with other derivatives methods
    # -----------------------------------------------------------------------
    #                       Symbolic differentiation
    # -----------------------------------------------------------------------
    x, y, z = sym.symbols('x y z')
    a_num, b_num, c_num = 2, 3, 6

    f_sym = (sym.sin(x) * sym.log(y)) + (sym.exp(y) * z) + (x * z)
    df_dx = sym.diff(f_sym, x)
    df_dy = sym.diff(f_sym, y)
    df_dz = sym.diff(f_sym, z)

    print('------------------------------------ \n'
            'Symbolic differentiation results:')
    print('Derivative of f(x, y, z) w.r.t x is: ', float(df_dx.subs([(x, a_num), (y, b_num), (z, c_num)])))
    print('Derivative of f(x, y, z) w.r.t y is: ', float(df_dy.subs([(x, a_num), (y, b_num), (z, c_num)])))
    print('Derivative of f(x, y, z) w.r.t z is: ', float(df_dz.subs([(x, a_num), (y, b_num), (z, c_num)])))
    print('------------------------------------\n')


    def f(x, y, z):
        """
        Function definition that will be used to test different differentiation methods
        """
        return (np.sin(x) * np.log(y)) + (np.exp(y) * z) + (x * z)

    # -----------------------------------------------------------------------
    #             Numerical differentiation (Finite difference)
    # -----------------------------------------------------------------------
    h = 1e-6
    df_dx = (f(a_num + h, b_num, c_num) - f(a_num, b_num, c_num)) / h
    df_dy = (f(a_num, b_num + h, c_num) - f(a_num, b_num, c_num)) / h
    df_dz = (f(a_num, b_num, c_num + h) - f(a_num, b_num, c_num)) / h

    print('-------------------------------------------- \n'
            'Numerical differentiation (Finite difference):')
    print('Derivative of f(x, y, z) w.r.t x is: ', df_dx)
    print('Derivative of f(x, y, z) w.r.t y is: ', df_dy)
    print('Derivative of f(x, y, z) w.r.t z is: ', df_dz)
    print('--------------------------------------------\n')

    # -----------------------------------------------------------------------
    #             Numerical differentiation (Complex step)
    # -----------------------------------------------------------------------
    h = 1e-6
    df_dx = (f(a_num + h * 1j, b_num, c_num).imag) / h
    df_dy = (f(a_num, b_num + h * 1j, c_num).imag) / h
    df_dz = (f(a_num, b_num, c_num + h * 1j).imag) / h

    print('-------------------------------------------- \n'
            'Numerical differentiation (Complex step):')
    print('Derivative of f(x, y, z) w.r.t x is: ', df_dx)
    print('Derivative of f(x, y, z) w.r.t y is: ', df_dy)
    print('Derivative of f(x, y, z) w.r.t z is: ', df_dz)
    print('--------------------------------------------\n')

