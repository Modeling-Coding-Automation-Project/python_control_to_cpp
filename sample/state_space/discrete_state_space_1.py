import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import control
import matplotlib.pyplot as plt

from python_control.state_space_deploy import StateSpaceDeploy
from python_control.simulation_plotter import SimulationPlotter

# define state-space model
A = np.array([[0.7, 0.2],
              [-0.3, 0.8]])
B = np.array([[0.1],
              [0.2]])
C = np.array([[2, 0]])
D = np.array([[0]])

dt = 0.01
sys = control.ss(A, B, C, D, dt)

# You can create cpp header which can easily define state space as C++ code
deployed_file_names = StateSpaceDeploy.generate_state_space_cpp_code(sys)
print(deployed_file_names)

# initialize state and input
x = np.array([[0],
              [0]])
u = 1  # input
n_steps = 50  # number of steps

plotter = SimulationPlotter()

# simulation
for _ in range(n_steps):
    y = C @ x + D * u
    x = A @ x + B * u

    plotter.append(x)
    plotter.append(y)

plotter.assign("x", column=0, row=0, position=(0, 0))
plotter.assign("x", column=1, row=0, position=(0, 0))
plotter.assign("y", column=0, row=0, position=(1, 0))

plotter.plot("Discrete-Time State-Space Response")
