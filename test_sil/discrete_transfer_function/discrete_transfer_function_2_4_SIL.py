import os
import sys
sys.path.append(os.getcwd())

import control
import numpy as np
import matplotlib.pyplot as plt

from python_control.transfer_function_deploy import TransferFunctionDeploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester

# Define discrete transfer function
sys_d = control.TransferFunction([0.5, 0.3, 0.1], [
    1.0, -1.8, 1.5, -0.7, 0.2], dt=0.2)

deployed_file_names = TransferFunctionDeploy.generate_transfer_function_cpp_code(
    sys_d, number_of_delay=0)

current_dir = os.path.dirname(__file__)
generator = SIL_CodeGenerator(deployed_file_names, current_dir)
generator.build_SIL_code()

from test_sil.discrete_transfer_function import DiscreteTransferFunctionSIL
DiscreteTransferFunctionSIL.initialize()

# simulation
T, yout = control.step_response(sys_d)

tester = MCAPTester()
NEAR_LIMIT = 1e-5

for i, time in enumerate(T):
    u = 1.0
    y = yout[i]

    U = np.zeros((1, 1))
    U[0, 0] = u
    DiscreteTransferFunctionSIL.update(np.array(U))

    y = DiscreteTransferFunctionSIL.get_y()

    tester.expect_near(y[0, 0], yout[i], NEAR_LIMIT,
                       f"Discrete transfer function 2 4 SIL, check Y at time {time:.2f}.")

tester.throw_error_if_test_failed()
