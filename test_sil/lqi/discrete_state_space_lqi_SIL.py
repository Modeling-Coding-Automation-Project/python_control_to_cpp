"""
Linear-Quadratic Integral (LQI) controller sample code - SIL test

Reference URL:
https://github.com/AtsushiSakai/PyAdvancedControl
https://jp.mathworks.com/help/control/ref/lti.lqr.html
https://jp.mathworks.com/help/control/ref/ss.lqi.html
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import control

from python_control.lqr_deploy import LQI_Deploy, LQR_METHOD
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester

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

# Create Expanded State Space Model
Ac_ex = np.block([
    [Ac, np.zeros((Ac.shape[0], Cc.shape[0]))],
    [Cc, np.zeros((Cc.shape[0], Cc.shape[0]))]])
Bc_ex = np.vstack([Bc, np.zeros((Cc.shape[0], Bc.shape[1]))])

# Discretize the continuous time model
sys_d = control.c2d(control.ss(Ac, Bc, Cc, Dc), dt, method='euler')
Ad = sys_d.A
Bd = sys_d.B
Cd = sys_d.C
Dd = sys_d.D

# LQI parameters
Q_ex = np.diag([1.0, 0.1, 1.0, 0.1, 2.0, 0.1])
R_ex = np.diag([1.0])

deployed_file_names = LQI_Deploy.generate_LQI_cpp_code(
    A=Ad,
    B=Bd,
    C=Cc,
    Q_ex=Q_ex,
    R_ex=R_ex,
    method=LQR_METHOD.DARE
)

current_dir = os.path.dirname(__file__)
generator = SIL_CodeGenerator(deployed_file_names, current_dir)
generator.build_SIL_code()

from test_sil.lqi import LqiSIL
LqiSIL.initialize()


def solve_dare_iterative(A, B, Q, R, max_iter=1000, tol=1e-10):
    P = Q.copy()

    for i in range(max_iter):
        BT_P = B.T @ P
        S = R + BT_P @ B

        K = np.linalg.solve(S, BT_P @ A)

        P_next = A.T @ P @ A - A.T @ P @ B @ K + Q

        err = np.linalg.norm(P_next - P, ord="fro")

        P = P_next

        if err < tol:
            return P, i + 1, True

    return P, max_iter, False


def dlqr_iterative(A, B, Q, R, max_iter=1000, tol=1e-10):
    P, num_iter, converged = solve_dare_iterative(
        A, B, Q, R, max_iter=max_iter, tol=tol
    )

    K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)

    return K


# Create Expanded Discrete State Space Model for DARE
Ad_ex = np.block([
    [Ad, np.zeros((Ad.shape[0], Cd.shape[0]))],
    [Cd, np.zeros((Cd.shape[0], Cd.shape[0]))]])
Bd_ex = np.vstack([Bd, np.zeros((Cd.shape[0], Bd.shape[1]))])

tester = MCAPTester()
NEAR_LIMIT = 1e-3

K_ex = dlqr_iterative(np.asarray(Ad_ex), np.asarray(Bd_ex), Q_ex, R_ex)
K_ex = np.matrix(K_ex)

K_cpp = LqiSIL.solve()

tester.expect_near(K_cpp, K_ex, NEAR_LIMIT,
                   "LQI SIL, check K.")

tester.throw_error_if_test_failed()
