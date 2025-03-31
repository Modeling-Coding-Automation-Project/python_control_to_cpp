"""
Linear-Quadratic Regulator sample code

Reference URL:
https://github.com/AtsushiSakai/PyAdvancedControl
https://jp.mathworks.com/help/control/ref/lti.lqr.html
https://jp.mathworks.com/help/control/ref/ss.lqi.html
"""
import os
import sys
sys.path.append(os.getcwd())

import control
import numpy as np
import scipy.linalg as la

from python_control.lqr_deploy import LQI_Deploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_sil.MCAP_tester import MCAPTester

simulation_time = 10.0
dt = 0.1

sys = control.TransferFunction([1.0], [1.0, 2.0, 1.0])
sys_ss = control.tf2ss(sys)

Ac = sys_ss.A
Bc = sys_ss.B
Cc = sys_ss.C
Dc = sys_ss.D

# Create Expanded State Space Model
Ac_ex = np.block([
    [Ac, np.zeros((Ac.shape[0], Cc.shape[0]))],
    [Cc, np.zeros((Cc.shape[0], Cc.shape[0]))]])
Bc_ex = np.vstack([Bc, np.zeros((Cc.shape[0], Bc.shape[1]))])

sys_ss_d = control.c2d(sys_ss, dt, method='euler')

Ad = sys_ss_d.A
Bd = sys_ss_d.B
Cd = sys_ss_d.C
Dd = sys_ss_d.D

# LQI parameters
Q_ex = np.diag([0.0, 2.0, 2.0])
R_ex = np.diag([1.0])

deployed_file_names = LQI_Deploy.generate_LQI_cpp_code(Ac, Bc, Cc, Q_ex, R_ex)

current_dir = os.path.dirname(__file__)
generator = SIL_CodeGenerator(deployed_file_names, current_dir)
generator.build_SIL_code()

from test_sil.lqi import LqiSIL
LqiSIL.initialize()


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
NEAR_LIMIT = 0.3

K_ex = lqr_with_arimoto_potter(Ac_ex, Bc_ex, Q_ex, R_ex)

LqiSIL.set_Eigen_solver_iteration_max(3)
LqiSIL.set_Eigen_solver_iteration_max_for_eigen_vector(8)
K_cpp = LqiSIL.solve()

tester.expect_near(K_cpp, K_ex, NEAR_LIMIT,
                   "LQI SIL, check K.")

tester.throw_error_if_test_failed()
