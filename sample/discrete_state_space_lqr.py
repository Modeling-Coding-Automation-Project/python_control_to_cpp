"""
Linear-Quadratic Regulator sample code

Reference URL:
https://github.com/AtsushiSakai/PyAdvancedControl
https://jp.mathworks.com/help/control/ref/lti.lqr.html
"""

import time
import matplotlib.pyplot as plt
import control
import numpy as np
import scipy.linalg as la

simulation_time = 10.0
dt = 0.1

# pendulum model continuous
Ac = np.matrix([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, -0.1, 3.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, -0.5, 30.0, 0.0]
])

Bc = np.matrix([
    [0.0],
    [2.0],
    [0.0],
    [5.0]
])

Cc = np.matrix([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

Dc = np.matrix([
    [0.0],
    [0.0]
])

# Discretize the continuous time model
sys_d = control.c2d(control.ss(Ac, Bc, Cc, Dc), dt, method='euler')
A = sys_d.A
B = sys_d.B
C = sys_d.C
D = sys_d.D

# LQR parameters
Q = np.diag([1.0, 0.0, 1.0, 0.0])
R = np.diag([1.0])
K_opt = None


def process(x, u):
    x = A * x + B * u
    return (x)


def solve_DARE_with_iteration(A, B, Q, R):
    """
    solve a discrete time_Algebraic Riccati equation (DARE)
    """
    X = Q
    maxiter = 150
    eps = 0.01

    for i in range(maxiter):
        Xn = A.T * X * A - A.T * X * B * \
            la.inv(R + B.T * X * B) * B.T * X * A + Q
        if (abs(Xn - X)).max() < eps:
            X = Xn
            break
        X = Xn

    return Xn


def lqr_with_arimoto_potter(Ac, Bc, Q, R):
    n = len(Bc)

    # Hamiltonian
    Hamiltonian = np.vstack(
        (np.hstack((Ac, - Bc * la.inv(R) * Bc.T)),
         np.hstack((-Q, -Ac.T))))

    eigen_values, eigen_vectors = la.eig(Hamiltonian)

    result = Hamiltonian @ eigen_vectors - \
        eigen_vectors @ np.diag(eigen_values)

    V1 = None
    V2 = None

    for i in range(2 * n):
        if eigen_values[i].real < 0:
            if V1 is None:
                V1 = eigen_vectors[0:n, i]
                V2 = eigen_vectors[n:2 * n, i]
            else:
                V1 = np.vstack((V1, eigen_vectors[0:n, i]))
                V2 = np.vstack((V2, eigen_vectors[n:2 * n, i]))
    V1 = np.matrix(V1.T)
    V2 = np.matrix(V2.T)

    P = (V2 * la.inv(V1)).real

    K = la.inv(R) * Bc.T * P

    return K


def dlqr_origin(Ad, Bd, Q, R):
    """
    This is the original dlqr function from Python Control Library
    """

    N = np.zeros((Q.shape[0], R.shape[1]))

    # Solve the discrete-time algebraic Riccati equation
    X = la.solve_discrete_are(Ad, Bd, Q, R, e=None, s=N)

    # Compute the LQR gain
    K = np.linalg.solve(Bd.T @ X @ Bd + R, Bd.T @ X @ Ad)

    return K


def lqr_ref_tracking(x, xref):
    global K_opt
    if K_opt is None:
        K_opt_arimoto_potter = lqr_with_arimoto_potter(Ac, Bc, Q, R)
        K_opt_origin_d = dlqr_origin(A, B, Q, R)
        K_opt_lqr, _, _ = control.lqr(Ac, Bc, Q, R)
        K_opt_dlqr, _, _ = control.dlqr(A, B, Q, R)

        print("K_opt_arimoto_potter:")
        print(K_opt_arimoto_potter)
        print("K_opt_origin_d:")
        print(K_opt_origin_d)
        print("K_opt_lqr:")
        print(K_opt_lqr)
        print("K_opt_dlqr:")
        print(K_opt_dlqr)

        K_opt = K_opt_arimoto_potter
        # K_opt = K_opt_dlqr

    u = K_opt * (xref - x)

    return u


def main_reference_tracking():
    t = 0.0

    x = np.matrix([
        [0.0],
        [0.0],
        [0.0],
        [0.0]
    ])
    u = np.matrix([0])

    xref = np.matrix([
        [1.0],
        [0.0],
        [0.0],
        [0.0]
    ])

    time_history = [0.0]
    x1_history = [x[0, 0]]
    x2_history = [x[2, 0]]
    u_history = [0.0]

    while t <= simulation_time:
        u = lqr_ref_tracking(x, xref)
        u0 = float(u[0, 0])

        x = process(x, u0)

        x1_history.append(x[0, 0])
        x2_history.append(x[2, 0])

        u_history.append(u0)
        time_history.append(t)
        t += dt

    # plot
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    fig.suptitle("LQR Tracking")

    ax1.grid(True)
    ax1.plot(time_history, x1_history, "-b", label="x")
    ax1.plot(time_history, x2_history, "-g", label="theta")
    xref0_h = [xref[0, 0] for i in range(len(time_history))]
    xref1_h = [xref[2, 0] for i in range(len(time_history))]
    ax1.plot(time_history, xref0_h, "--b", label="target x")
    ax1.plot(time_history, xref1_h, "--g", label="target theta")
    ax1.legend()

    ax2.plot(time_history, u_history, "-r", label="input")
    ax2.grid(True)
    ax2.legend()

    plt.show()


if __name__ == '__main__':
    main_reference_tracking()
