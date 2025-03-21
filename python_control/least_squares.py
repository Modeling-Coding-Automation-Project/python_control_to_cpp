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
