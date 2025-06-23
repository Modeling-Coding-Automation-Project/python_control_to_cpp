"""
File: least_squares_method.py

This script demonstrates the use of the LeastSquares and LeastSquaresDeploy classes
for performing and deploying a linear least squares regression model.
It generates synthetic data with two features and an offset, fits a linear model,
predicts outputs, and compares the learned weights to the true weights.
The script also shows how to generate C++ header files for the learned model and visualizes the input data.
"""
import os

import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import matplotlib.pyplot as plt

from external_libraries.MCAP_python_control.python_control.least_squares import LeastSquares
from python_control.least_squares_deploy import LeastSquaresDeploy

# Create data
np.random.seed(42)
n_samples = 20
x1 = np.random.rand(n_samples) * 10
x2 = np.random.rand(n_samples) * 10
offset = np.random.normal(0.3, 0.01, n_samples)

y = 1.5 * x1 - 0.8 * x2 + offset
X = np.column_stack((x1, x2))

ls = LeastSquares(X)

# You can create cpp header which can easily define least squares as C++ code
deployed_file_names = LeastSquaresDeploy.generate_LS_cpp_code(ls)
print(deployed_file_names)

# Learn the model
ls.fit(X, y)
predictions = ls.predict(X)

print("true weights:", [1.5, -0.8, 0.3])
print("predicted weights:", ls.get_weights())
print("true y:", y[:5])
print("predicted y:", predictions[:5])

# plot
fig, axs = plt.subplots(3, 1)
axs[0].plot(x1, label="x1")
axs[0].legend()
axs[0].grid(True)
axs[1].plot(x2, label="x2")
axs[1].legend()
axs[1].grid(True)
axs[2].plot(offset, label="offset")
axs[2].legend()
axs[2].grid(True)

plt.show()
