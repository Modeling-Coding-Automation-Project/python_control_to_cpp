import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import control
import matplotlib.pyplot as plt

from python_control.state_space_deploy import StateSpaceDeploy


# define state-space model
A = np.array([[0.7, 0.2],
              [-0.3, 0.8]])
B = np.array([[0.1],
              [0.2]])
C = np.array([[2, 0]])
D = np.array([[0]])

# You can create cpp header which can easily define state space as C++ code
dt = 0.01
sys = control.ss(A, B, C, D, dt)
deployed_file_names = StateSpaceDeploy.generate_state_space_cpp_code(sys)
print(deployed_file_names)

# initialize state and input
x = np.array([[0],
              [0]])
u = 1  # input
n_steps = 50  # number of steps

# history
x_history = []
y_history = []

# simulation
for _ in range(n_steps):
    y = C @ x + D * u
    x = A @ x + B * u

    x_history.append(x.flatten())
    y_history.append(y.flatten()[0])

# plot results
x_history = np.array(x_history)
y_history = np.array(y_history)

print("\nx_history:\n", x_history)
print("\n\ny_history:\n", y_history)

plt.figure(figsize=(10, 6))
plt.plot(range(n_steps), x_history[:, 0], label="x1[k] (State 1)")
plt.plot(range(n_steps), x_history[:, 1], label="x2[k] (State 2)")
plt.plot(range(n_steps), y_history, label="y[k] (Output)")
plt.xlabel("Time step k")
plt.ylabel("Value")
plt.title("Discrete-Time State-Space Response")
plt.legend()
plt.grid()
plt.show()
