import os
import sys
sys.path.append(os.getcwd())

import numpy as np

from python_control.kalman_filter import LinearKalmanFilter
from python_control.kalman_filter_deploy import KalmanFilterDeploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester


def generate_m_sequence(length, taps):
    """Generate an M-sequence using a linear feedback shift register (LFSR)."""
    state = [1] * (max(taps) + 1)  # Initial state with all ones
    m_sequence = []

    for _ in range(length):
        m_sequence.append(state[-1])
        feedback = sum([state[tap] for tap in taps]) % 2
        state = [feedback] + state[:-1]

    return np.array(m_sequence)


if __name__ == "__main__":
    # Create model
    A = np.array([[1.0, 0.1, 0.0, 0.0],
                  [0.0, 1.0, 0.1, 0.0],
                  [0.0, 0.0, 1.0, 0.1],
                  [0.0, 0.0, 0.0, 1.0]])

    B = np.array([[0.0, 0.0],
                  [0.1, 0.0],
                  [0.0, 0.1],
                  [0.0, 0.0]])

    C = np.array([[1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]])

    Number_of_Delay = 2

    # System noise and observation noise parameters
    Q = np.diag([1.0, 1.0, 1.0, 2.0])
    R = np.eye(2) * 10.0

    # Define Kalman filter
    lkf = LinearKalmanFilter(A, B, C, Q, R, Number_of_Delay)

    deployed_file_names = KalmanFilterDeploy.generate_LKF_cpp_code(
        lkf, number_of_delay=Number_of_Delay)

    current_dir = os.path.dirname(__file__)
    generator = SIL_CodeGenerator(deployed_file_names, current_dir)
    generator.build_SIL_code()

    from test_sil.kalman_filter import KalmanFilterSIL
    KalmanFilterSIL.initialize()

    # Initial state
    lkf.x_hat = np.array([[0],
                         [0],
                         [0],
                         [0]])

    # Simulation steps
    num_steps = 50

    # Generate input signal
    taps = [2, 3]
    m_sequence = generate_m_sequence(num_steps * 2, taps)
    u_data_T = m_sequence.reshape(num_steps, 2) - 0.5
    u_data = u_data_T.T

    # System noise and observation noise real
    Q_real = np.diag([1.0, 1.0, 1.0, 1.0]) * 0.0
    R_real = np.eye(2) * 0.0

    # Generate data
    np.random.seed(0)

    x_true = np.array([[0.0], [0.0], [0.0], [0.1]])
    x_estimate = lkf.get_x_hat()
    y_measured = np.zeros((C.shape[0], 1))
    u = np.zeros((B.shape[1], 1))

    y_store = [np.zeros((C.shape[0], 1))] * (Number_of_Delay + 1)
    delay_index = 0

    x_cpp = np.array([[0.0], [0.0], [0.0], [0.0]])
    KalmanFilterSIL.set_x_hat(x_cpp)

    tester = MCAPTester()
    NEAR_LIMIT = 1e-5

    for k in range(1, num_steps):
        u = u_data[:, k - 1].reshape(-1, 1)

        w = np.random.multivariate_normal(np.zeros(A.shape[0]), Q_real)
        v = np.random.multivariate_normal(np.zeros(C.shape[0]), R_real)

        # system response
        x_true = A @ x_true + B @ u + w.reshape(-1, 1)
        y_store[delay_index] = C @ x_true + v.reshape(-1, 1)

        # system delay
        delay_index += 1
        if delay_index > Number_of_Delay:
            delay_index = 0

        y_measured = y_store[delay_index]

        # Kalman filter
        lkf.predict_and_update(u, y_measured)
        x_estimate = lkf.get_x_hat()

        KalmanFilterSIL.predict_and_update(u, y_measured)
        x_estimate_cpp = KalmanFilterSIL.get_x_hat()

        tester.expect_near(x_estimate_cpp, x_estimate, NEAR_LIMIT,
                           "Linear Kalman Filter SIL, check x_hat.")

tester.throw_error_if_test_failed()
