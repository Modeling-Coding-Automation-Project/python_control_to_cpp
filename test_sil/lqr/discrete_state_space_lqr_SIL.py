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
from test_sil.SIL_operator import SIL_CodeGenerator
from test_sil.MCAP_tester import MCAPTester

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


deployed_file_names = LQR_Deploy.generate_LQR_cpp_code(Ac, Bc, Q, R)

current_dir = os.path.dirname(__file__)
generator = SIL_CodeGenerator(deployed_file_names, current_dir)
generator.build_SIL_code()

from test_sil.lqr import LqrSIL
LqrSIL.initialize()


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


tester = MCAPTester()
NEAR_LIMIT = 1e-5

K_python = lqr_with_arimoto_potter(Ac, Bc, Q, R)
K_cpp = LqrSIL.solve()

tester.expect_near(K_cpp, K_python, NEAR_LIMIT,
                   "LQR SIL, check K.")

tester.throw_error_if_test_failed()
