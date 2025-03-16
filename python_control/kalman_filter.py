import copy
import math
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


class KalmanFilterCommon:
    def __init__(self, Number_of_Delay=0):
        self.Number_of_Delay = Number_of_Delay

    def predict_and_update(self, u, y):
        self.predict(u)
        self.update(y)

    def get_x_hat(self):
        return self.x_hat


class LinearKalmanFilter(KalmanFilterCommon):
    def __init__(self, A, B, C, Q, R, Number_of_Delay=0):
        super().__init__(Number_of_Delay)
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x_hat = np.zeros((A.shape[0], 1))
        self.P = np.eye(A.shape[0])
        self.G = None

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

    # If G is known, you can use below "_fixed_G" functions.
    def predict_with_fixed_G(self, u):
        self.x_hat = self.A @ self.x_hat + self.B @ u

    def update_with_fixed_G(self, y):
        self.x_hat = self.x_hat + self.G @ self.calc_y_dif(y)

    def predict_and_update_with_fixed_G(self, u, y):
        self.predict_with_fixed_G(u)
        self.update_with_fixed_G(y)


class ExtendedKalmanFilter(KalmanFilterCommon):
    def __init__(self, state_function, measurement_function,
                 state_function_jacobian, measurement_function_jacobian,
                 Q, R, Parameters=None, Number_of_Delay=0):
        super().__init__(Number_of_Delay)
        self.state_function = state_function
        self.measurement_function = measurement_function
        self.state_function_jacobian = state_function_jacobian
        self.measurement_function_jacobian = measurement_function_jacobian

        self.A = np.zeros(Q.shape)
        self.C = np.zeros((R.shape[0], Q.shape[0]))
        self.Q = Q
        self.R = R

        self.x_hat = np.zeros((Q.shape[0], 1))
        self.P = np.eye(Q.shape[0])
        self.G = None

        self.Parameters = Parameters
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
        # y_dif = y - self.measurement_function(self.x_hat, self.Parameters)

        return y_dif


class UnscentedKalmanFilter_Basic(KalmanFilterCommon):
    def __init__(self, state_function, measurement_function,
                 Q, R, Parameters=None, Number_of_Delay=0, kappa=0.0):
        super().__init__(Number_of_Delay)
        self.state_function = state_function
        self.measurement_function = measurement_function

        self.Q = Q
        self.R = R

        self.STATE_SIZE = Q.shape[0]
        self.OUTPUT_SIZE = R.shape[0]

        self.x_hat = np.zeros((self.STATE_SIZE, 1))
        self.X_d = np.zeros((self.STATE_SIZE, 2 * self.STATE_SIZE + 1))

        self.P = np.eye(self.STATE_SIZE)
        self.G = None

        self.Parameters = Parameters
        self.y_store = DelayedVectorObject(R.shape[0], Number_of_Delay)

        self.kappa = kappa
        self.lambda_weight = 0.0

        self.W = np.zeros((2 * self.STATE_SIZE + 1, 2 * self.STATE_SIZE + 1))

        self.calc_weights()

    def calc_weights(self):

        self.lambda_weight = self.kappa

        self.W[0, 0] = self.kappa / \
            (self.STATE_SIZE + self.kappa)
        for i in range(1, 2 * self.STATE_SIZE + 1):
            self.W[i, i] = 1 / (2 * (self.STATE_SIZE + self.kappa))

    def calc_sigma_points(self, x, P):
        SP = np.linalg.cholesky(P)
        Kai = np.zeros((self.STATE_SIZE, 2 * self.STATE_SIZE + 1))

        Kai[:, 0] = x.flatten()
        for i in range(self.STATE_SIZE):
            Kai[:, i + 1] = (x +
                             math.sqrt(self.STATE_SIZE + self.lambda_weight) *
                             (SP[:, i]).reshape(-1, 1)).flatten()
            Kai[:, i + self.STATE_SIZE + 1] = \
                (x -
                 math.sqrt(self.STATE_SIZE + self.lambda_weight) *
                 (SP[:, i]).reshape(-1, 1)).flatten()

        return Kai

    def predict(self, u):
        Kai = self.calc_sigma_points(self.x_hat, self.P)

        for i in range(2 * self.STATE_SIZE + 1):
            Kai[:, i] = self.state_function(
                Kai[:, i].reshape(-1, 1), u, self.Parameters).flatten()

        self.x_hat = np.zeros((self.STATE_SIZE, 1))
        for i in range(2 * self.STATE_SIZE + 1):
            self.x_hat += self.W[i, i] * Kai[:, i].reshape(-1, 1)

        for i in range(2 * self.STATE_SIZE + 1):
            self.X_d[:, i] = (Kai[:, i].reshape(-1, 1) - self.x_hat).flatten()

        self.P = self.X_d @ self.W @ self.X_d.T + self.Q

    def update(self, y):
        Kai = self.calc_sigma_points(self.x_hat, self.P)

        Nu = np.zeros((self.OUTPUT_SIZE, 2 * self.STATE_SIZE + 1))
        for i in range(2 * self.STATE_SIZE + 1):
            Nu[:, i] = self.measurement_function(
                Kai[:, i].reshape(-1, 1), self.Parameters).flatten()

        y_hat_m = np.zeros((self.OUTPUT_SIZE, 1))
        for i in range(2 * self.STATE_SIZE + 1):
            y_hat_m += self.W[i, i] * Nu[:, i].reshape(-1, 1)

        Y_d = np.zeros((self.OUTPUT_SIZE, 2 * self.STATE_SIZE + 1))
        for i in range(2 * self.STATE_SIZE + 1):
            Y_d[:, i] = (Nu[:, i].reshape(-1, 1) - y_hat_m).flatten()

        P_yy = Y_d @ self.W @ Y_d.T
        P_xy = self.X_d @ self.W @ Y_d.T

        self.G = P_xy @ np.linalg.inv(P_yy + self.R)

        self.x_hat = self.x_hat + self.G @ self.calc_y_dif(y, y_hat_m)
        self.P = self.P - self.G @ P_xy.T

    def calc_y_dif(self, y, y_hat_m):
        self.y_store.push(y_hat_m)

        y_dif = y - self.y_store.get()

        # When there is no delay, you can use below.
        # y_dif = y - y_hat_m

        return y_dif


class UnscentedKalmanFilter(UnscentedKalmanFilter_Basic):
    def __init__(self, state_function, measurement_function,
                 Q, R, Parameters=None, Number_of_Delay=0, kappa=0.0, alpha=0.5, beta=2.0):

        self.kappa = 0.0
        if kappa == 0.0:
            self.kappa = 3 - Q.shape[0]
        else:
            self.kappa = kappa

        self.alpha = alpha
        self.beta = beta
        self.lambda_weight = 0.0

        self.Wc = np.zeros((2 * Q.shape[0] + 1, 2 * Q.shape[0] + 1))
        self.wc = 0.0

        super().__init__(state_function, measurement_function,
                         Q, R, Parameters, Number_of_Delay, self.kappa)

    def calc_weights(self):
        self.lambda_weight = self.alpha * self.alpha * \
            (self.STATE_SIZE + self.kappa) - self.STATE_SIZE

        self.W[0, 0] = self.lambda_weight / \
            (self.STATE_SIZE + self.lambda_weight)
        for i in range(1, 2 * self.STATE_SIZE + 1):
            self.W[i, i] = 1 / (2 * (self.STATE_SIZE + self.lambda_weight))

        self.wc = self.W[0, 0] + 1 - self.alpha * self.alpha + self.beta

        self.Wc = copy.deepcopy(self.W)
        self.Wc[0, 0] = self.wc

    def predict(self, u):
        Kai = self.calc_sigma_points(self.x_hat, self.P)

        for i in range(2 * self.STATE_SIZE + 1):
            Kai[:, i] = self.state_function(
                Kai[:, i].reshape(-1, 1), u, self.Parameters).flatten()

        self.x_hat = np.zeros((self.STATE_SIZE, 1))
        for i in range(2 * self.STATE_SIZE + 1):
            self.x_hat += self.W[i, i] * Kai[:, i].reshape(-1, 1)

        for i in range(2 * self.STATE_SIZE + 1):
            self.X_d[:, i] = (Kai[:, i].reshape(-1, 1) - self.x_hat).flatten()

        self.P = self.X_d @ self.Wc @ self.X_d.T + self.Q

    def update(self, y):
        Kai = self.calc_sigma_points(self.x_hat, self.P)

        Nu = np.zeros((self.OUTPUT_SIZE, 2 * self.STATE_SIZE + 1))
        for i in range(2 * self.STATE_SIZE + 1):
            Nu[:, i] = self.measurement_function(
                Kai[:, i].reshape(-1, 1), self.Parameters).flatten()

        y_hat_m = np.zeros((self.OUTPUT_SIZE, 1))
        for i in range(2 * self.STATE_SIZE + 1):
            y_hat_m += self.W[i, i] * Nu[:, i].reshape(-1, 1)

        Y_d = np.zeros((self.OUTPUT_SIZE, 2 * self.STATE_SIZE + 1))
        for i in range(2 * self.STATE_SIZE + 1):
            Y_d[:, i] = (Nu[:, i].reshape(-1, 1) - y_hat_m).flatten()

        P_yy = Y_d @ self.Wc @ Y_d.T
        P_xy = self.X_d @ self.Wc @ Y_d.T

        self.G = P_xy @ np.linalg.inv(P_yy + self.R)

        self.x_hat = self.x_hat + self.G @ self.calc_y_dif(y, y_hat_m)
        self.P = self.P - self.G @ P_xy.T
