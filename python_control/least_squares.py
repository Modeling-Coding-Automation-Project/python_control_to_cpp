import numpy as np


class LeastSquares:
    def __init__(self, X):
        self.state_size = X.shape[1]
        if self.state_size <= 0:
            raise ValueError("State size must be greater than 0.")

        self.number_of_data = X.shape[0]

        self.weights = np.zeros((self.state_size + 1, 1), dtype=X.dtype)

    def fit(self, X, y):

        if X.shape[1] != self.state_size:
            raise ValueError("Wrong state size.")

        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term

        self.weights = np.linalg.lstsq(X, y)[0]

    def predict(self, X):

        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term

        return X @ self.weights

    def get_weights(self):
        return self.weights


class RecursiveLeastSquares:
    def __init__(self, feature_size, lambda_factor=0.9, delta=0.1):
        self.RLS_size = feature_size + 1  # Add bias term
        self.lambda_factor = lambda_factor
        self.delta = delta
        self.P = np.eye(self.RLS_size) * delta
        self.weights = np.zeros((self.RLS_size, 1))

    def update(self, x, y_true):
        Ones = np.ones((1, 1))  # Add bias term
        x_ex = np.vstack((x, Ones))

        y = x_ex.T @ self.weights
        y_dif = y_true - y

        P_x = self.P @ x_ex
        K = P_x / (self.lambda_factor + x_ex.T @ P_x)

        self.weights += K @ y_dif
        self.P = (self.P - K @ x_ex.T @ self.P) / self.lambda_factor

    def predict(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack((X, ones)) @ self.weights

    def get_weights(self):
        return self.weights
