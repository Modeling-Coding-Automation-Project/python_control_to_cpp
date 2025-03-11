import numpy as np


class DelayedVectorObject:
    def __init__(self, size, Number_of_Delay):
        self.store = np.zeros((size, Number_of_Delay + 1))
        self.delay_index = 0
        self.Number_of_Delay = Number_of_Delay

    def push(self, vector):
        self.store[:, self.delay_index] = vector.flatten()

        self.delay_index += 1
        if self.delay_index > self.Number_of_Delay:
            self.delay_index = 0

    def get(self):
        return self.store[:, self.delay_index].reshape(-1, 1)


class LinearKalmanFilter:
    def __init__(self, A, B, C, Q, R, Number_of_Delay=0):
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x_hat = np.zeros((A.shape[0], 1))
        self.P = np.ones(A.shape)
        self.G = None

        self.Number_of_Delay = Number_of_Delay
        self.y_store = DelayedVectorObject(C.shape[0], Number_of_Delay)

    def predict(self, u):
        self.x_hat = self.A @ self.x_hat + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        P_CT = self.P @ self.C.T

        S = self.C @ P_CT + self.R
        self.G = P_CT @ np.linalg.inv(S)
        self.x_hat = self.x_hat + self.G @ self.calc_y_dif(y)

        self.P = (np.eye(self.A.shape[0]) - self.G @ self.C) @ self.P

    def calc_y_dif(self, y):
        self.y_store.push(self.C @ self.x_hat)

        y_dif = y - self.y_store.get()

        # When there is no delay, you can use below.
        # y_dif = y - self.C @ self.x_hat

        return y_dif

    def predict_and_update(self, u, y):
        self.predict(u)
        self.update(y)

    # If G is known, you can use below "_fixed_G" functions.
    def predict_with_fixed_G(self, u):
        self.x_hat = self.A @ self.x_hat + self.B @ u

    def update_with_fixed_G(self, y):
        self.x_hat = self.x_hat + self.G @ self.calc_y_dif(y)

    def predict_and_update_with_fixed_G(self, u, y):
        self.predict_with_fixed_G(u)
        self.update_with_fixed_G(y)

    def get_x_hat(self):
        return self.x_hat


class ExtendedKalmanFilter:
    def __init__(self, state_function, measurement_function,
                 state_function_jacobian, measurement_function_jacobian,
                 Q, R, Parameters=None, Number_of_Delay=0):
        self.state_function = state_function
        self.measurement_function = measurement_function
        self.state_function_jacobian = state_function_jacobian
        self.measurement_function_jacobian = measurement_function_jacobian

        self.A = np.zeros(Q.shape[0])
        self.C = np.zeros((R.shape[0], Q.shape[0]))
        self.Q = Q
        self.R = R

        self.x_hat = np.zeros((Q.shape[0], 1))
        self.P = np.ones(Q.shape)
        self.G = None

        self.Parameters = Parameters
        self.Number_of_Delay = Number_of_Delay
        self.y_store = DelayedVectorObject(R.shape[0], Number_of_Delay)

    def predict(self, u):
        self.A = self.state_function_jacobian(self.x_hat, u, self.Parameters)
        self.x_hat = self.state_function(self.x_hat, u, self.Parameters)
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        self.C = self.measurement_function_jacobian(
            self.x_hat, self.Parameters)

        P_CT = self.P @ self.C.T

        S = self.C @ P_CT + self.R
        self.G = P_CT @ np.linalg.inv(S)
        self.x_hat = self.x_hat + self.G @ self.calc_y_dif(y)

        self.P = (np.eye(self.A.shape[0]) - self.G @ self.C) @ self.P

    def calc_y_dif(self, y):
        self.y_store.push(self.measurement_function(
            self.x_hat, self.Parameters))

        y_dif = y - self.y_store.get()

        # When there is no delay, you can use below.
        # y_dif = y - self.measurement_function(self.x_hat)

        return y_dif

    def predict_and_update(self, u, y):
        self.predict(u)
        self.update(y)

    def get_x_hat(self):
        return self.x_hat
