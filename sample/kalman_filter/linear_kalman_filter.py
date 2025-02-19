import matplotlib.pyplot as plt
import numpy as np


class LinearKalmanFilter:
    def __init__(self, A, B, C, Q, R):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x_hat = np.zeros((A.shape[0], 1))
        self.P = np.ones(A.shape[0])

    def predict(self, u):
        self.x_hat = self.A @ self.x_hat + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        P_CT = self.P @ self.C.T

        S = self.C @ P_CT + self.R
        G = P_CT @ np.linalg.inv(S)
        self.x_hat = self.x_hat + G @ self.calc_y_dif(y)
        self.P = (np.eye(A.shape[0]) - G @ self.C) @ self.P

    def calc_y_dif(self, y):
        y_dif = y - self.C @ self.x_hat
        return y_dif

    def get_x_hat(self):
        return self.x_hat


# Example
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

    # System noise and observation noise
    Q = np.eye(4) * 0.1
    R = np.eye(2) * 10.0

    # Define Kalman filter
    kf = LinearKalmanFilter(A, B, C, Q, R)

    # Initial state
    kf.x_hat = np.array([[0],
                         [0],
                         [0],
                         [0]])

    # Simulation steps
    num_steps = 100

    # Generate input signal
    taps = [2, 3]
    m_sequence = generate_m_sequence(num_steps * 2, taps)
    u_data = m_sequence.reshape(num_steps, 2) - 0.5

    # Generate data
    np.random.seed(0)

    x_data = np.zeros((num_steps, A.shape[0]))
    y_data = np.zeros((num_steps, C.shape[0]))
    x_data[0] = kf.x_hat.T

    for k in range(1, num_steps):
        w = np.random.multivariate_normal(np.zeros(4), Q)
        v = np.random.multivariate_normal(np.zeros(2), R)
        x_data[k] = A @ x_data[k-1] + B @ u_data[k-1] + w
        y_data[k] = C @ x_data[k] + v

    # Plot
    plt.figure()
    plt.title("True state and observation")

    plt.subplot(3, 2, 1)
    plt.plot(x_data[:, 0], label="True x0")
    plt.plot(y_data[:, 0], label="Observation y0")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.subplot(3, 2, 3)
    plt.plot(x_data[:, 2], label="True x2")
    plt.plot(y_data[:, 1], label="Observation y1")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.subplot(3, 2, 2)
    plt.plot(u_data[:, 0], label="Input u0")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.subplot(3, 2, 4)
    plt.plot(u_data[:, 1], label="Input u1")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.subplot(3, 2, 5)
    plt.plot(x_data[:, 1], label="True x1")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.subplot(3, 2, 6)
    plt.plot(x_data[:, 3], label="True x3")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Value")

    plt.show()
