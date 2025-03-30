import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import control
import matplotlib.pyplot as plt

from test_sil.discrete_state_space import DiscreteStateSpaceSIL

# define state-space model
A = np.array([[0.7, 0.2],
              [-0.3, 0.8]])
B = np.array([[0.1],
              [0.2]])
C = np.array([[2, 0]])
D = np.array([[0]])

dt = 0.01
sys = control.ss(A, B, C, D, dt)

DiscreteStateSpaceSIL.initialize()

# initialize state and input
x = np.array([[0],
              [0]])
u = 1  # input
n_steps = 50  # number of steps

# simulation
for _ in range(n_steps):
    y = C @ x + D * u
    x = A @ x + B * u

    U = np.zeros((1, 1))
    U[0, 0] = u
    DiscreteStateSpaceSIL.update(np.array(U))

    X = DiscreteStateSpaceSIL.get_X()
    Y = DiscreteStateSpaceSIL.get_Y()

    print("X_0:", X[0, 0], ", ", x[0, 0])
    print("X_1:", X[1, 0], ", ", x[1, 0])
    print("Y:", Y[0], ", ", y[0, 0])
    print("\n")
