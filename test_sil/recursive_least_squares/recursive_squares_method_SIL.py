import os
import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import copy
import matplotlib.pyplot as plt

from python_control.least_squares import RecursiveLeastSquares
from python_control.least_squares_deploy import LeastSquaresDeploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester

# Create data
np.random.seed(42)
n_samples = 100
x1 = np.random.rand(n_samples) * 10
x2 = np.random.rand(n_samples) * 10
offset = np.random.normal(0.3, 0.01, n_samples)

weights_true = np.array([[0.5], [-0.2], [0.3]])

y = weights_true[0, 0] * x1 + weights_true[1, 0] * x2 + offset
X = np.column_stack((x1, x2))

# Create Recursive Least Squares object
rls = RecursiveLeastSquares(feature_size=X.shape[1], lambda_factor=0.9)

deployed_file_names = LeastSquaresDeploy.generate_RLS_cpp_code(rls)

current_dir = os.path.dirname(__file__)
generator = SIL_CodeGenerator(deployed_file_names, current_dir)
generator.build_SIL_code()

from test_sil.recursive_least_squares import RecursiveLeastSquaresSIL
RecursiveLeastSquaresSIL.initialize()

tester = MCAPTester()
NEAR_LIMIT = 1e-5

for i in range(n_samples):
    x = X[i].reshape(-1, 1)
    y_true = y[i].reshape(-1, 1)

    rls.update(x, y_true)

    weights_predicted = copy.deepcopy(rls.get_weights())

    RecursiveLeastSquaresSIL.update(x, y_true)
    weights_predicted_cpp = RecursiveLeastSquaresSIL.get_weights()

    tester.expect_near(weights_predicted_cpp, weights_predicted, NEAR_LIMIT,
                       "Recursive Least Squares SIL, check weights.")

tester.throw_error_if_test_failed()
