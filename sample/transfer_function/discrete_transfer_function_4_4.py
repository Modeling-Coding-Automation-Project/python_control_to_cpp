"""
File: discrete_transfer_function_4_4.py

This script demonstrates the creation, analysis, and deployment of
a discrete transfer function using the `control` library in Python.
It defines a discrete transfer function, generates corresponding
C++ header code for deployment, and visualizes the step response.
The script also converts the transfer function to a discrete state-space
representation and compares the step responses of both forms.
"""
import os

import sys
sys.path.append(os.getcwd())

import control
import numpy as np
import matplotlib.pyplot as plt

from python_control.transfer_function_deploy import TransferFunctionDeploy


# Define continuous transfer function
sys_d = control.TransferFunction([1.0, 0.5, 0.3, 0.2, 0.1], [
    1.1, -0.5, 0.4, -0.3, 0.2], dt=0.2)
print("\nDiscrete transfer function:\n", sys_d)

# You can create cpp header which can easily define transfer function as C++ code
deployed_file_names = TransferFunctionDeploy.generate_transfer_function_cpp_code(
    sys_d, number_of_delay=0)
print(deployed_file_names, "\n")

# step response
T, yout = control.step_response(sys_d)

# plot results
plt.plot(T, yout)
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.title('Step Response')
plt.grid(True)

# convert to discrete state space
sys_ss_d = control.tf2ss(sys_d)
T_ss_d, yout_ss_d = control.step_response(sys_ss_d)

# plot results
plt.plot(T_ss_d, yout_ss_d)
plt.legend(['continuous', 'discrete'])

# show y results
print("yout_ss_d")
for i in range(len(yout_ss_d)):
    print(yout_ss_d[i], ",")

# show results
plt.show()
