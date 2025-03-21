import os
import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import copy
import matplotlib.pyplot as plt

from python_control.least_squares import RecursiveLeastSquares
from python_control.simulation_plotter import SimulationPlotter

# Create data
np.random.seed(42)
n_samples = 100
x1 = np.random.rand(n_samples) * 10
x2 = np.random.rand(n_samples) * 10
offset = np.random.normal(0.3, 0.01, n_samples)

weights_true = np.array([[0.5], [-0.2], [0.3]])

y = weights_true[0, 0] * x1 + weights_true[1, 0] * x2 + offset
X = np.column_stack((x1, x2))

# Learn the model
model = RecursiveLeastSquares(feature_size=2, lambda_factor=0.9)

plotter = SimulationPlotter()

for i in range(n_samples):
    x = X[i].reshape(-1, 1)
    y_true = y[i].reshape(-1, 1)

    model.update(x, y_true)

    weights_predicted = copy.deepcopy(model.get_weights())
    plotter.append(weights_predicted)
    plotter.append(weights_true)

predictions = model.predict(X)

print("true weights:", weights_true.T)
print("predicted weights:", model.get_weights().T)
print("true y:", y[:5])
print("predicted y:", predictions[:5].T)

# Plot the results
plotter.assign("weights_predicted", column=0, row=0, position=(0, 0))
plotter.assign("weights_true", column=0, row=0, position=(0, 0))
plotter.assign("weights_predicted", column=1, row=0, position=(1, 0))
plotter.assign("weights_true", column=1, row=0, position=(1, 0))
plotter.assign("weights_predicted", column=2, row=0, position=(2, 0))
plotter.assign("weights_true", column=2, row=0, position=(2, 0))

plotter.plot("Discrete-Time State-Space Response")
