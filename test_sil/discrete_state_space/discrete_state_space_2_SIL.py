from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
import control
import matplotlib.pyplot as plt

from python_control.state_space_deploy import StateSpaceDeploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester

# define continuous state-space model
A = np.array([
    [0.0, 1.0, 0.0, 0.0],
    [-51.2077076817131, -1.0, 2.56038538408566, 0.0],
    [0.0, 0.0, 0.0, 1.0],
    [128.019900633784, 0.0, -6.40099503168920, -10.2]
])

B = np.array([
    [0.0],
    [0.0],
    [0.0],
    [1.0]
])
C = np.array([
    [1, 0, 0, 0],
    [1280.19900633784, 0, -64.0099503168920, 0]
])
D = np.array([
    [0],
    [0]
])

sys = control.StateSpace(A, B, C, D)

# convert to discrete state-space model
dt = 0.01
sys_d = sys.sample(Ts=dt, method='euler')
T_d, yout_d = control.step_response(sys_d)

# You can create cpp header which can easily define state space as C++ code
deployed_file_names = StateSpaceDeploy.generate_state_space_cpp_code(
    sys_d, number_of_delay=0)

current_dir = os.path.dirname(__file__)
generator = SIL_CodeGenerator(deployed_file_names, current_dir)
generator.build_SIL_code()

from test_sil.discrete_state_space import DiscreteStateSpaceSIL
DiscreteStateSpaceSIL.initialize()

# simulation
tester = MCAPTester()
NEAR_LIMIT = 1e-5

u = 1.0  # input

for i, t in enumerate(T_d):

    U = np.zeros((1, 1))
    U[0, 0] = u
    DiscreteStateSpaceSIL.update(np.array(U))

    Y = DiscreteStateSpaceSIL.get_Y()

    tester.expect_near(Y[0, 0], yout_d[0, 0, i], NEAR_LIMIT,
                       "Discrete state space 2 SIL, check Y_0.")
    tester.expect_near(Y[1, 0], yout_d[1, 0, i], NEAR_LIMIT,
                       "Discrete state space 2 SIL, check Y_1.")

tester.throw_error_if_test_failed()
