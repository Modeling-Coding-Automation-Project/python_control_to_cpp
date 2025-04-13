import os
import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import matplotlib.pyplot as plt

from python_control.least_squares import LeastSquares
from python_control.least_squares_deploy import LeastSquaresDeploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester

# Create data
np.random.seed(42)
n_samples = 20
x1 = np.random.rand(n_samples) * 10
x2 = np.random.rand(n_samples) * 10
offset = np.random.normal(0.3, 0.01, n_samples)

y = 1.5 * x1 - 0.8 * x2 + offset
y = y.reshape((n_samples, 1))
X = np.column_stack((x1, x2))

ls = LeastSquares(X)

deployed_file_names = LeastSquaresDeploy.generate_LS_cpp_code(ls)

current_dir = os.path.dirname(__file__)
generator = SIL_CodeGenerator(deployed_file_names, current_dir)
generator.build_SIL_code()

from test_sil.least_squares import LeastSquaresSIL
LeastSquaresSIL.initialize()

# Learn the model
ls.fit(X, y)
weights = ls.get_weights()

LeastSquaresSIL.fit(X, y)
weights_cpp = LeastSquaresSIL.get_weights()

tester = MCAPTester()
NEAR_LIMIT = 1e-5

tester.expect_near(weights_cpp, weights, NEAR_LIMIT,
                   "Least Squares SIL, check weights.")

tester.throw_error_if_test_failed()
