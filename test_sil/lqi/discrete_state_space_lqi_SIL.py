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
from scipy.linalg import solve_discrete_are

from python_control.lqr_deploy import LQI_Deploy, LQR_METHOD
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester

simulation_time = 10.0
dt = 0.1

# pendulum model continuous
Ac = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [0.0, -0.1, 3.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [0.0, -0.5, 30.0, 0.0]
])

Bc = np.array([
    [0.0],
    [2.0],
    [0.0],
    [5.0]
])

Cc = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0]
])

Dc = np.array([
    [0.0],
    [0.0]
])

# Integrate cart position so the augmented plant is stabilizable.
C_integral = Cc[:1, :]
integral_size = C_integral.shape[0]

# Discretize the continuous time model
sys_d = control.c2d(control.ss(Ac, Bc, Cc, Dc), dt, method='euler')
Ad = sys_d.A
Bd = sys_d.B

# Match the discrete LQI sample's weights.
Q_ex = np.diag([0.3, 0.01, 1.0, 10.0, 0.08])
R_ex = np.diag([1.0])

deployed_file_names = LQI_Deploy.generate_LQI_cpp_code(
    A=Ad,
    B=Bd,
    C=C_integral,
    Q_ex=Q_ex,
    R_ex=R_ex,
    method=LQR_METHOD.DARE,
    dt=dt
)

current_dir = os.path.dirname(__file__)
generator = SIL_CodeGenerator(deployed_file_names, current_dir)
generator.build_SIL_code()

from test_sil.lqi import LqiSIL
LqiSIL.initialize()


# Use SciPy's DARE solution as an independent reference.
Ad_ex = np.block([
    [Ad, np.zeros((Ad.shape[0], integral_size))],
    [dt * C_integral, np.eye(integral_size)]])
Bd_ex = np.vstack([Bd, np.zeros((integral_size, Bd.shape[1]))])
P = solve_discrete_are(Ad_ex, Bd_ex, Q_ex, R_ex)
BT_P = Bd_ex.T @ P
K_ex = np.linalg.solve(R_ex + BT_P @ Bd_ex, BT_P @ Ad_ex)

tester = MCAPTester()
K_cpp = LqiSIL.solve()
tester.expect_near(K_cpp, K_ex, 1e-6, "LQI SIL, check DARE gain.")
np.testing.assert_array_equal(LqiSIL.get_K(), K_cpp,
                              err_msg="LQI SIL, check stored gain.")

closed_loop_poles = np.linalg.eigvals(Ad_ex - Bd_ex @ K_cpp)
if np.any(np.abs(closed_loop_poles) >= 1.0):
    raise AssertionError("LQI SIL, the discrete closed loop is unstable.")

# Track the sample's position command with the same constant input offset.
state_size = Ad.shape[0]
K_x = K_cpp[:, :state_size]
K_e = K_cpp[:, state_size:]
x_ref = np.array([[1.0], [0.0], [0.0], [0.0]])
y_ref = C_integral @ x_ref
x = np.zeros_like(x_ref)
e_y_integral = np.zeros_like(y_ref)
u_offset = 0.1

for _ in range(round(simulation_time / dt)):
    u = K_x @ (x_ref - x) + K_e @ e_y_integral
    e_y_integral += dt * (y_ref - C_integral @ x)
    x = Ad @ x + Bd @ (u + u_offset)

tester.expect_near(Cc @ x, Cc @ x_ref, 1e-3,
                   "LQI SIL, check position and angle with input offset.")
tester.throw_error_if_test_failed()
