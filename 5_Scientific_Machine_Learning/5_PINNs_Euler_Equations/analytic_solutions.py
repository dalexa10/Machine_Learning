import numpy as np
from scipy.optimize import newton
import matplotlib.pyplot as plt

def set_boundary_conditions_Sod_problem(nx):
    """
    Set the boundary conditions for the Sod shock tube problem
    :param nx: (float) number of points in the discretization
    :return:
        U_l (list): left state vector [density, velocity, pressure]
        U_r (list): right state vector [density, velocity, pressure]
        x0 (int): initial position of the discontinuity (default middle point)
    """
    
    U_l = [1.0, 0.0, 1]     # Left state
    U_r = [0.125, 0.0, .1]  # Right state
    x0 = nx // 2            # Initial position of discontinuity
    
    return U_l, U_r, x0
    
    
def f(P, pL, pR, cL, cR, gam):
    """ 
    Residual function to be solved with Newton's method
    """

    a = (gam - 1) * (cR / cL) * (P - 1)
    b = np.sqrt(2 * gam * (2 * gam + (gam + 1) * (P - 1)))

    return P - pL / pR * (1 - a / b) ** (2. * gam / (gam - 1.))


def SodShockAnalytic(x, T, gam=7/5):
    """
    Analytic solution to the Sod shock tube problem
    :param x: (np.array) space discretization
    :param T: (float) final time for numerical simulation
    :param gam: (float) adiabatic constant 1.4=7/5 for a 3D diatomic gas 
    :return: 
        U_analytic (np.array): analytic solution to the Sod shock tube problem
                    np.array([[rho], [u], [p]])
    """
    # Preprocessing 
    nx = len(x)
    dx = x[1] - x[0]
    
    # Set boundary conditions
    U_l, U_r, x0 = set_boundary_conditions_Sod_problem(nx)
    rho_l, u_l, p_l = U_l
    rho_r, u_r, p_r = U_r

    U_analytic = np.zeros((3, nx), dtype='float64')

    # compute speed of sound
    cL = np.sqrt(gam * p_l / rho_l)
    cR = np.sqrt(gam * p_r / rho_r)
    # compute P
    P = newton(f, 0.5, args=(p_l, p_r, cL, cR, gam), tol=1e-12)

    # compute region positions right to left
    # region R
    c_shock = u_r + cR * np.sqrt((gam - 1 + P * (gam + 1)) / (2 * gam))
    x_shock = x0 + int(np.floor(c_shock * T / dx))
    U_analytic[0, x_shock - 1:] = rho_r
    U_analytic[1, x_shock - 1:] = u_r
    U_analytic[2, x_shock - 1:] = p_r

    # region 2
    alpha = (gam + 1) / (gam - 1)
    c_contact = u_l + 2 * cL / (gam - 1) * (1 - (P * p_r / p_l) ** ((gam - 1.) / 2 / gam))
    x_contact = x0 + int(np.floor(c_contact * T / dx))
    U_analytic[0, x_contact:x_shock - 1] = (1 + alpha * P) / (alpha + P) * rho_r
    U_analytic[1, x_contact:x_shock - 1] = c_contact
    U_analytic[2, x_contact:x_shock - 1] = P * p_r

    # region 3
    r3 = rho_l * (P * p_r / p_l) ** (1 / gam)
    p3 = P * p_r
    c_fanright = c_contact - np.sqrt(gam * p3 / r3)
    x_fanright = x0 + int(np.ceil(c_fanright * T / dx))
    U_analytic[0, x_fanright:x_contact] = r3
    U_analytic[1, x_fanright:x_contact] = c_contact
    U_analytic[2, x_fanright:x_contact] = P * p_r

    # region 4
    c_fanleft = -cL
    x_fanleft = x0 + int(np.ceil(c_fanleft * T / dx))
    u4 = 2 / (gam + 1) * (cL + (x[x_fanleft:x_fanright] - x[x0]) / T)
    U_analytic[0, x_fanleft:x_fanright] = rho_l * (1 - (gam - 1) / 2. * u4 / cL) ** (2 / (gam - 1))
    U_analytic[1, x_fanleft:x_fanright] = u4
    U_analytic[2, x_fanleft:x_fanright] = p_l * (1 - (gam - 1) / 2. * u4 / cL) ** (2 * gam / (gam - 1))

    # region L
    U_analytic[0, :x_fanleft] = rho_l
    U_analytic[1, :x_fanleft] = u_l
    U_analytic[2, :x_fanleft] = p_l

    return U_analytic

if __name__ == '__main__':

    # Set Discretization
    Nx = 100
    X = 1.
    dx = X / (Nx - 1)
    x_vec = np.linspace(0, X, Nx)
    T = 0.2

    U_analytic = SodShockAnalytic(x_vec, T)

    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].set_title("Density")
    ax[0].plot(x_vec, U_analytic[0].T)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel(r'$\rho$')

    ax[1].set_title("Flow speed")
    ax[1].plot(x_vec, U_analytic[1].T)
    ax[1].set_xlabel('x')
    ax[1].set_ylabel(r'$u$')

    ax[2].set_title("Pressure")
    ax[2].plot(x_vec, U_analytic[2].T)
    for i in range(3):
        ax[i].set_xlim([0., 1.])
        ax[i].set_ylim([-.05, 1.05])
    ax[2].set_xlabel('x')
    ax[2].set_ylabel(r'$p$')

    plt.tight_layout()
    plt.show()