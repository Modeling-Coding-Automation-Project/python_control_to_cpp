import numpy as np


class LinearKalmanFilter:
    def __init__(self, A, B, C, Q, R, Number_of_Delay=0):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x_hat = np.zeros((A.shape[0], 1))
        self.P = np.ones(A.shape[0])
        self.Number_of_Delay = Number_of_Delay
        self.y_store = np.zeros((C.shape[0], Number_of_Delay + 1))
        self.delay_index = 0

    def predict(self, u):
        self.x_hat = self.A @ self.x_hat + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        P_CT = self.P @ self.C.T

        S = self.C @ P_CT + self.R
        G = P_CT @ np.linalg.inv(S)
        self.x_hat = self.x_hat + G @ self.calc_y_dif(y)
        self.P = (np.eye(self.A.shape[0]) - G @ self.C) @ self.P

    def calc_y_dif(self, y):
        self.y_store[:, self.delay_index] = (self.C @ self.x_hat
                                             ).flatten()

        # update delay index
        self.delay_index += 1
        if self.delay_index > self.Number_of_Delay:
            self.delay_index = 0

        y_dif = y - self.y_store[:, self.delay_index].reshape(-1, 1)
        return y_dif

    def get_x_hat(self):
        return self.x_hat
