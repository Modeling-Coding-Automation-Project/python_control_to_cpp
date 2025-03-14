import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt

from python_control.kalman_filter import LinearKalmanFilter
from python_control.kalman_filter_deploy import KalmanFilterDeploy


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

    # You can create cpp header which can easily define state space as C++ code
    deployed_file_names = KalmanFilterDeploy.generate_LKF_cpp_code(lkf)
    print(deployed_file_names)

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

    x_true = [np.zeros((A.shape[0], 1))] * num_steps
    x_true[0] = np.array([[0.0], [0.0], [0.0], [0.1]])

    x_estimate = [np.zeros((A.shape[0], 1))] * num_steps
    x_estimate[0] = lkf.get_x_hat()

    y_measured = [np.zeros((C.shape[0], 1))] * num_steps

    y_store = [np.zeros((C.shape[0], 1))] * (Number_of_Delay + 1)
    delay_index = 0

    for k in range(1, num_steps):
        u = u_data[:, k - 1].reshape(-1, 1)

        w = np.random.multivariate_normal(np.zeros(A.shape[0]), Q_real)
        v = np.random.multivariate_normal(np.zeros(C.shape[0]), R_real)

        # system response
        x_true[k] = A @ x_true[k - 1] + B @ u + w.reshape(-1, 1)
        y_store[delay_index] = C @ x_true[k] + v.reshape(-1, 1)

        # system delay
        delay_index += 1
        if delay_index > Number_of_Delay:
            delay_index = 0

        y_measured[k] = y_store[delay_index]

        # Kalman filter
        lkf.predict_and_update(u, y_measured[k])
        x_estimate[k] = lkf.get_x_hat()

    # Kalman Gain
    print("Kalman Gain:\n", lkf.G)

    # Plot
    fig, axs = plt.subplots(3, 2)
    fig.suptitle("True state and observation")

    axs[0, 0].plot([x[0, 0] for x in x_true], label="True x0")
    axs[0, 0].plot([x[0, 0] for x in x_estimate], label="Estimated x0")
    axs[0, 0].plot([x[0, 0] for x in y_measured], label="Measured y0")
    axs[0, 0].legend()
    axs[0, 0].set_ylabel("Value")
    axs[0, 0].grid(True)

    axs[1, 0].plot([x[2, 0] for x in x_true], label="True x2")
    axs[1, 0].plot([x[2, 0] for x in x_estimate], label="Estimated x2")
    axs[1, 0].plot([x[1, 0] for x in y_measured], label="Measured y1")
    axs[1, 0].legend()
    axs[1, 0].set_ylabel("Value")
    axs[1, 0].grid(True)

    axs[0, 1].plot(u_data[0, :], label="Input u0")
    axs[0, 1].legend()
    axs[0, 1].set_ylabel("Value")
    axs[0, 1].grid(True)

    axs[1, 1].plot(u_data[1, :], label="Input u1")
    axs[1, 1].legend()
    axs[1, 1].set_ylabel("Value")
    axs[1, 1].grid(True)

    axs[2, 0].plot([x[1, 0] for x in x_true], label="True x1")
    axs[2, 0].plot([x[1, 0] for x in x_estimate], label="Estimated x1")
    axs[2, 0].legend()
    axs[2, 0].set_xlabel("Time")
    axs[2, 0].set_ylabel("Value")
    axs[2, 0].grid(True)

    axs[2, 1].plot([x[3, 0] for x in x_true], label="True x3")
    axs[2, 1].plot([x[3, 0] for x in x_estimate], label="Estimated x3")
    axs[2, 1].legend()
    axs[2, 1].set_xlabel("Time")
    axs[2, 1].set_ylabel("Value")
    axs[2, 1].grid(True)

    plt.show()
