import os
import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import matplotlib.pyplot as plt

from python_control.least_squares import RecursiveLeastSquares

# Create data
np.random.seed(42)
n_samples = 20
x1 = np.random.rand(n_samples) * 10
x2 = np.random.rand(n_samples) * 10
offset = np.random.normal(0.3, 0.01, n_samples)

y = 1.5 * x1 - 0.8 * x2 + offset
X = np.column_stack((x1, x2))

# Learn the model
model = RecursiveLeastSquares(feature_size=2)

for i in range(n_samples):
    x = X[i].reshape(-1, 1)
    y_true = y[i].reshape(-1, 1)

    model.update(x, y_true)

predictions = model.predict(X)

print("true weights:", [1.5, -0.8, 0.3])
print("predicted weights:", model.get_weights())
print("true y:", y[:5])
print("predicted y:", predictions[:5])
