"""
Linear-Quadratic Regulator sample code

Reference URL:
https://github.com/AtsushiSakai/PyAdvancedControl
https://jp.mathworks.com/help/control/ref/lti.lqr.html
"""
import os
import sys
sys.path.append(os.getcwd())

import matplotlib.pyplot as plt
import control
import numpy as np
import scipy.linalg as la

from python_control.lqr_deploy import LQR_Deploy

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
Ad = sys_d.A
Bd = sys_d.B
Cd = sys_d.C
Dd = sys_d.D

# LQR parameters
Q = np.diag([1.0, 0.0, 1.0, 0.0])
R = np.diag([1.0])

# You can create cpp header which can easily define state space as C++ code
deployed_file_names = LQR_Deploy.generate_LQR_cpp_code(Ac, Bc, Q, R)
print(deployed_file_names)


def process(x, u):
    x = Ad * x + Bd * u
    return (x)


def lqr_with_arimoto_potter(Ac, Bc, Q, R):
    n = len(Bc)

    # Hamiltonian
    Hamiltonian = np.vstack(
        (np.hstack((Ac, - Bc * la.inv(R) * Bc.T)),
         np.hstack((-Q, -Ac.T))))

    eigen_values, eigen_vectors = la.eig(Hamiltonian)

    V1 = None
    V2 = None

    minus_count = 0
    for i in range(2 * n):
        if eigen_values[i].real < 0:
            if V1 is None:
                V1 = eigen_vectors[0:n, i]
                V2 = eigen_vectors[n:2 * n, i]
            else:
                V1 = np.vstack((V1, eigen_vectors[0:n, i]))
                V2 = np.vstack((V2, eigen_vectors[n:2 * n, i]))

            minus_count += 1
            if minus_count == n:
                break

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


def main_reference_tracking():
    # design LQR controller
    K = lqr_with_arimoto_potter(Ac, Bc, Q, R)
    # K, _, _ = control.lqr(Ac, Bc, Q, R)
    # K = dlqr_origin(A, B, Q, R)
    # K, _, _ = control.dlqr(A, B, Q, R)

    print("K: ")
    print(K)

    # prepare simulation
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

    # simulation
    while t <= simulation_time:
        u = K * (xref - x)
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
