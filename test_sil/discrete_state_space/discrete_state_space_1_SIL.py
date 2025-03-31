import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import control
import matplotlib.pyplot as plt

from python_control.state_space_deploy import StateSpaceDeploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_sil.MCAP_tester import MCAPTester

# define state-space model
A = np.array([[0.7, 0.2],
              [-0.3, 0.8]])
B = np.array([[0.1],
              [0.2]])
C = np.array([[2.0, 0.0]])
D = np.array([[0.0]])

dt = 0.01
sys = control.ss(A, B, C, D, dt)

deployed_file_names = StateSpaceDeploy.generate_state_space_cpp_code(
    sys, number_of_delay=0)

current_dir = os.path.dirname(__file__)
generator = SIL_CodeGenerator(deployed_file_names, current_dir)
generator.build_SIL_code()

from test_sil.discrete_state_space import DiscreteStateSpaceSIL
DiscreteStateSpaceSIL.initialize()

# initialize state and input
x = np.array([[0.0],
              [0.0]])
u = 1.0  # input
n_steps = 50  # number of steps

# simulation
tester = MCAPTester()
NEAR_LIMIT = 1e-5

for _ in range(n_steps):
    y = C @ x + D * u
    x = A @ x + B * u

    U = np.zeros((1, 1))
    U[0, 0] = u
    DiscreteStateSpaceSIL.update(np.array(U))

    X = DiscreteStateSpaceSIL.get_X()
    Y = DiscreteStateSpaceSIL.get_Y()

    tester.expect_near(X[0, 0], x[0, 0], NEAR_LIMIT,
                       "Discrete state space 1 SIL, check X_0.")
    tester.expect_near(X[1, 0], x[1, 0], NEAR_LIMIT,
                       "Discrete state space 1 SIL, check X_1.")
    tester.expect_near(Y[0, 0], y[0, 0], NEAR_LIMIT,
                       "Discrete state space 1 SIL, check Y.")

tester.throw_error_if_test_failed()
