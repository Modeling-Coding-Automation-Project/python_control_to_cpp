import os
import sys
sys.path.append(os.getcwd())

import numpy as np

from python_control.kalman_filter import LinearKalmanFilter
from python_control.kalman_filter_deploy import KalmanFilterDeploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester


def main():
    # Create model
    A = np.array([[0.7, 0.2],
                  [-0.3, 0.8]])
    B = np.array([[0.1],
                  [0.2]])
    C = np.array([[1.0, 0.0]])
    # D = np.array([[0.0]])

    dt = 0.01

    # System noise and observation noise parameters
    Q = np.diag([1.0, 1.0])
    R = np.diag([1.0])

    # Define Kalman filter
    lkf = LinearKalmanFilter(A, B, C, Q, R)

    # You can get converged G if you use LKF.
    lkf.converge_G()

    deployed_file_names = KalmanFilterDeploy.generate_LKF_cpp_code(lkf)

    current_dir = os.path.dirname(__file__)
    generator = SIL_CodeGenerator(deployed_file_names, current_dir)
    generator.build_SIL_code()

    from test_sil.kalman_filter_fix import KalmanFilterFixSIL
    KalmanFilterFixSIL.initialize()

    # Initial state
    lkf.x_hat = np.array([[0],
                         [0]])

    # Simulation steps
    num_steps = 100
    time = np.arange(0, num_steps * dt, dt)

    # System noise and observation noise real
    Q_real = np.diag([1.0, 1.0]) * 0.0
    R_real = np.diag([1.0]) * 0.0

    # Generate data
    np.random.seed(0)

    x_true = np.array([[0.1], [0.1]])
    x_estimate = lkf.get_x_hat()
    y_measured = np.zeros((C.shape[0], 1))
    u = np.ones((B.shape[1], 1))

    tester = MCAPTester()
    NEAR_LIMIT = 1e-5

    for k in range(1, num_steps):

        w = np.random.multivariate_normal(np.zeros(A.shape[0]), Q_real)
        v = np.random.multivariate_normal(np.zeros(C.shape[0]), R_real)

        # system response
        x_true = A @ x_true + B @ u + w.reshape(-1, 1)
        y_measured = C @ x_true + v.reshape(-1, 1)

        lkf.predict_and_update_with_fixed_G(u, y_measured)
        x_estimate = lkf.get_x_hat()

        KalmanFilterFixSIL.predict_and_update_with_fixed_G(u, y_measured)
        x_estimate_cpp = KalmanFilterFixSIL.get_x_hat()

        tester.expect_near(x_estimate_cpp, x_estimate, NEAR_LIMIT,
                           "Linear Kalman Filter with fixed G SIL, check x_hat.")

        u_latest = lkf.u_store.get_latest()
        u_latest_cpp = KalmanFilterFixSIL.u_store_get_latest()

        tester.expect_near(u_latest_cpp, u_latest, NEAR_LIMIT,
                           "Linear Kalman Filter with fixed G SIL, check u_store latest.")

    tester.throw_error_if_test_failed()


if __name__ == "__main__":
    main()
