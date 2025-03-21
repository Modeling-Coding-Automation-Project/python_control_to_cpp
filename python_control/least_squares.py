import numpy as np


class LeastSquares:
    def __init__(self):
        self.weights = None

    def fit(self, X, y):

        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term

        self.weights = np.linalg.lstsq(X, y)[0]

    def predict(self, X):

        if self.weights is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        X = np.hstack([X, np.ones((X.shape[0], 1))])  # Add bias term

        return X @ self.weights

    def get_weights(self):
        return self.weights


class RecursiveLeastSquares:
    def __init__(self, feature_size, lambda_factor=0.9, delta=0.1):
        self.RLS_size = feature_size + 1  # Add bias term
        self.lambda_factor = lambda_factor
        self.delta = delta
        self.P = np.eye(self.RLS_size) / delta
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
