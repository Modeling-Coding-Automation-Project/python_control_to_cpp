import math
import numpy as np
import copy

KALMAN_FILTER_DIVISION_MIN = 1e-10
LKF_G_CONVERGE_REPEAT_MAX = 1000


class DelayedVectorObject:
    def __init__(self, size, Number_of_Delay):
        self.store = np.zeros((size, Number_of_Delay + 1))
        self.delay_index = 0
        self.Number_of_Delay = Number_of_Delay

    def push(self, vector: np.ndarray):
        self.store[:, self.delay_index] = vector.flatten()

        self.delay_index += 1
        if self.delay_index > self.Number_of_Delay:
            self.delay_index = 0

    def get(self):
        return self.store[:, self.delay_index].reshape(-1, 1)

    def get_by_index(self, index):
        if index > self.Number_of_Delay:
            index = self.Number_of_Delay

        return self.store[:, index].reshape(-1, 1)


class KalmanFilterCommon:
    def __init__(self, Number_of_Delay=0):
        self.Number_of_Delay = Number_of_Delay
        self._input_count = 0

    def predict_and_update(self, u: np.ndarray, y: np.ndarray):
        self.predict(u)
        self.update(y)

    def get_x_hat(self):
        return self.x_hat


class LinearKalmanFilter(KalmanFilterCommon):
    def __init__(self, A: np.ndarray, B: np.ndarray,
                 C: np.ndarray, Q: np.ndarray, R: np.ndarray,
                 Number_of_Delay=0):
        super().__init__(Number_of_Delay)
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.R = R
        self.x_hat = np.zeros((A.shape[0], 1))
        self.P = np.eye(A.shape[0])
        self.G = None

        self.u_store = DelayedVectorObject(B.shape[1], Number_of_Delay)

    def predict(self, u: np.ndarray):
        self.u_store.push(u)

        if self._input_count < self.Number_of_Delay:
            self._input_count += 1
        else:
            self.x_hat = self.A @ self.x_hat + self.B @ self.u_store.get()
            self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y: np.ndarray):
        P_CT_matrix = self.P @ self.C.T

        S_matrix = self.C @ P_CT_matrix + self.R
        self.G = P_CT_matrix @ np.linalg.inv(S_matrix)
        self.x_hat = self.x_hat + self.G @ (y - self.C @ self.x_hat)

        self.P = (np.eye(self.A.shape[0]) - self.G @ self.C) @ self.P

    def get_x_hat_without_delay(self):
        if self.Number_of_Delay == 0:
            return self.x_hat
        else:
            x_hat = copy.deepcopy(self.x_hat)
            delay_index = self.u_store.delay_index + \
                self.Number_of_Delay - self._input_count

            for i in range(self._input_count):
                delay_index += 1
                if delay_index > self.Number_of_Delay:
                    delay_index = delay_index - self.Number_of_Delay - 1

                x_hat = self.A @ x_hat + \
                    self.B @ self.u_store.get_by_index(delay_index)

            return x_hat

    # If G is known, you can use below "_fixed_G" functions.
    def predict_with_fixed_G(self, u: np.ndarray):
        if self._input_count < self.Number_of_Delay:
            self._input_count += 1
        else:
            self.x_hat = self.A @ self.x_hat + self.B @ u

    def update_with_fixed_G(self, y: np.ndarray):
        self.x_hat = self.x_hat + self.G @ (y - self.C @ self.x_hat)

    def predict_and_update_with_fixed_G(self, u: np.ndarray, y: np.ndarray):
        self.predict_with_fixed_G(u)
        self.update_with_fixed_G(y)

    def update_P_one_step(self):
        self.P = self.A @ self.P @ self.A.T + self.Q

        P_CT_matrix = self.P @ self.C.T
        S_matrix = self.C @ P_CT_matrix + self.R
        self.G = P_CT_matrix @ np.linalg.inv(S_matrix)

        self.P = (np.eye(self.A.shape[0]) - self.G @ self.C) @ self.P

    def converge_G(self):

        for k in range(LKF_G_CONVERGE_REPEAT_MAX):
            if self.G is None:
                self.update_P_one_step()

            previous_G = self.G
            self.update_P_one_step()
            G_diff = self.G - previous_G

            is_converged = True
            for i in range(self.G.shape[0]):
                for j in range(self.G.shape[1]):

                    if (abs(self.G[i, j]) > KALMAN_FILTER_DIVISION_MIN) and \
                            (abs(G_diff[i, j] / self.G[i, j]) > KALMAN_FILTER_DIVISION_MIN):
                        is_converged = False

            if is_converged:
                break


class ExtendedKalmanFilter(KalmanFilterCommon):
    def __init__(self, state_function, measurement_function,
                 state_function_jacobian, measurement_function_jacobian,
                 Q: np.ndarray, R: np.ndarray, Parameters=None, Number_of_Delay=0):
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
        self.u_store = None

    def predict(self, u: np.ndarray):
        if self.u_store is None:
            self.u_store = DelayedVectorObject(
                u.shape[0], self.Number_of_Delay)

        self.u_store.push(u)

        if self._input_count < self.Number_of_Delay:
            self._input_count += 1
        else:
            self.A = self.state_function_jacobian(
                self.x_hat, self.u_store.get(), self.Parameters)
            self.x_hat = self.state_function(
                self.x_hat, self.u_store.get(), self.Parameters)
            self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, y):
        self.C = self.measurement_function_jacobian(
            self.x_hat, self.Parameters)

        P_CT_matrix = self.P @ self.C.T

        S_matrix = self.C @ P_CT_matrix + self.R
        self.G = P_CT_matrix @ np.linalg.inv(S_matrix)
        self.x_hat = self.x_hat + self.G @ (y - self.measurement_function(
            self.x_hat, self.Parameters))

        self.P = (np.eye(self.A.shape[0]) - self.G @ self.C) @ self.P

    def get_x_hat_without_delay(self):
        if self.Number_of_Delay == 0:
            return self.x_hat
        else:
            x_hat = copy.deepcopy(self.x_hat)
            delay_index = self.u_store.delay_index + \
                self.Number_of_Delay - self._input_count

            for i in range(self._input_count):
                delay_index += 1
                if delay_index > self.Number_of_Delay:
                    delay_index = delay_index - self.Number_of_Delay - 1

                x_hat = self.state_function(
                    x_hat, self.u_store.get_by_index(delay_index), self.Parameters)

            return x_hat


class UnscentedKalmanFilter_Basic(KalmanFilterCommon):
    def __init__(self, state_function, measurement_function,
                 Q: np.ndarray, R: np.ndarray, Parameters=None,
                 Number_of_Delay=0, kappa=0.0):
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
        self.u_store = None

        self.kappa = kappa
        self.sigma_point_weight = 0.0

        self.W = np.zeros((2 * self.STATE_SIZE + 1, 2 * self.STATE_SIZE + 1))

        self.calc_weights()

    def calc_weights(self):

        lambda_weight = self.kappa

        self.W[0, 0] = self.kappa / \
            (self.STATE_SIZE + self.kappa)
        for i in range(1, 2 * self.STATE_SIZE + 1):
            self.W[i, i] = 1.0 / (2.0 * (self.STATE_SIZE + self.kappa))

        self.sigma_point_weight = math.sqrt(self.STATE_SIZE + lambda_weight)

    def calc_sigma_points(self, x: np.ndarray, P: np.ndarray):
        sqrtP = np.linalg.cholesky(P, upper=True)
        Kai = np.zeros((self.STATE_SIZE, 2 * self.STATE_SIZE + 1))

        Kai[:, 0] = x.flatten()
        for i in range(self.STATE_SIZE):
            Kai[:, i + 1] = (x + self.sigma_point_weight *
                             (sqrtP[:, i]).reshape(-1, 1)).flatten()
            Kai[:, i + self.STATE_SIZE + 1] = \
                (x - self.sigma_point_weight *
                 (sqrtP[:, i]).reshape(-1, 1)).flatten()

        return Kai

    def predict(self, u: np.ndarray):
        if self.u_store is None:
            self.u_store = DelayedVectorObject(
                u.shape[0], self.Number_of_Delay)

        self.u_store.push(u)

        if self._input_count < self.Number_of_Delay:
            self._input_count += 1
        else:
            Kai = self.calc_sigma_points(self.x_hat, self.P)

            for i in range(2 * self.STATE_SIZE + 1):
                Kai[:, i] = self.state_function(
                    Kai[:, i].reshape(-1, 1), u, self.Parameters).flatten()

            self.x_hat = np.zeros((self.STATE_SIZE, 1))
            for i in range(2 * self.STATE_SIZE + 1):
                self.x_hat += self.W[i, i] * Kai[:, i].reshape(-1, 1)

            for i in range(2 * self.STATE_SIZE + 1):
                self.X_d[:, i] = (Kai[:, i].reshape(-1, 1) -
                                  self.x_hat).flatten()

            self.P = self.X_d @ self.W @ self.X_d.T + self.Q

    def update(self, y):
        Kai = self.calc_sigma_points(self.x_hat, self.P)

        Y_d = np.zeros((self.OUTPUT_SIZE, 2 * self.STATE_SIZE + 1))
        for i in range(2 * self.STATE_SIZE + 1):
            Y_d[:, i] = self.measurement_function(
                Kai[:, i].reshape(-1, 1), self.Parameters).flatten()

        y_hat_m = np.zeros((self.OUTPUT_SIZE, 1))
        for i in range(2 * self.STATE_SIZE + 1):
            y_hat_m += self.W[i, i] * Y_d[:, i].reshape(-1, 1)

        for i in range(2 * self.STATE_SIZE + 1):
            Y_d[:, i] = (Y_d[:, i].reshape(-1, 1) - y_hat_m).flatten()

        P_yy = Y_d @ self.W @ Y_d.T
        P_xy = self.X_d @ self.W @ Y_d.T

        self.G = P_xy @ np.linalg.inv(P_yy + self.R)

        self.x_hat = self.x_hat + self.G @ (y - y_hat_m)
        self.P = self.P - self.G @ P_xy.T

    def get_x_hat_without_delay(self):
        if self.Number_of_Delay == 0:
            return self.x_hat
        else:
            x_hat = copy.deepcopy(self.x_hat)
            delay_index = self.u_store.delay_index + \
                self.Number_of_Delay - self._input_count

            for i in range(self._input_count):
                delay_index += 1
                if delay_index > self.Number_of_Delay:
                    delay_index = delay_index - self.Number_of_Delay - 1

                x_hat = self.state_function(
                    x_hat, self.u_store.get_by_index(delay_index), self.Parameters)

            return x_hat


class UnscentedKalmanFilter(UnscentedKalmanFilter_Basic):
    def __init__(self, state_function, measurement_function,
                 Q: np.ndarray, R: np.ndarray, Parameters=None,
                 Number_of_Delay=0, kappa=0.0, alpha=0.5, beta=2.0):

        self.kappa = 0.0
        if kappa == 0.0:
            self.kappa = 3 - Q.shape[0]
        else:
            self.kappa = kappa

        self.alpha = alpha
        self.beta = beta
        self.sigma_point_weight = 0.0

        self.w_m = 0.0

        super().__init__(state_function, measurement_function,
                         Q, R, Parameters, Number_of_Delay, self.kappa)

    def calc_weights(self):
        lambda_weight = self.alpha * self.alpha * \
            (self.STATE_SIZE + self.kappa) - self.STATE_SIZE

        self.w_m = lambda_weight / \
            (self.STATE_SIZE + lambda_weight)
        self.W[0, 0] = self.w_m + 1.0 - self.alpha * self.alpha + self.beta
        for i in range(1, 2 * self.STATE_SIZE + 1):
            self.W[i, i] = 1.0 / (2.0 * (self.STATE_SIZE + lambda_weight))

        self.sigma_point_weight = math.sqrt(self.STATE_SIZE + lambda_weight)

    def predict(self, u):
        if self.u_store is None:
            self.u_store = DelayedVectorObject(
                u.shape[0], self.Number_of_Delay)

        self.u_store.push(u)

        if self._input_count < self.Number_of_Delay:
            self._input_count += 1
        else:
            Kai = self.calc_sigma_points(self.x_hat, self.P)

            for i in range(2 * self.STATE_SIZE + 1):
                Kai[:, i] = self.state_function(
                    Kai[:, i].reshape(-1, 1), u, self.Parameters).flatten()

            self.x_hat = self.w_m * Kai[:, 0].reshape(-1, 1)
            for i in range(1, 2 * self.STATE_SIZE + 1):
                self.x_hat += self.W[i, i] * Kai[:, i].reshape(-1, 1)

            for i in range(2 * self.STATE_SIZE + 1):
                self.X_d[:, i] = (Kai[:, i].reshape(-1, 1) -
                                  self.x_hat).flatten()

            self.P = self.X_d @ self.W @ self.X_d.T + self.Q

    def update(self, y):
        Kai = self.calc_sigma_points(self.x_hat, self.P)

        Y_d = np.zeros((self.OUTPUT_SIZE, 2 * self.STATE_SIZE + 1))
        for i in range(2 * self.STATE_SIZE + 1):
            Y_d[:, i] = self.measurement_function(
                Kai[:, i].reshape(-1, 1), self.Parameters).flatten()

        y_hat_m = self.w_m * Y_d[:, 0].reshape(-1, 1)
        for i in range(1, 2 * self.STATE_SIZE + 1):
            y_hat_m += self.W[i, i] * Y_d[:, i].reshape(-1, 1)

        for i in range(2 * self.STATE_SIZE + 1):
            Y_d[:, i] = (Y_d[:, i].reshape(-1, 1) - y_hat_m).flatten()

        P_yy = Y_d @ self.W @ Y_d.T
        P_xy = self.X_d @ self.W @ Y_d.T

        self.G = P_xy @ np.linalg.inv(P_yy + self.R)

        self.x_hat = self.x_hat + self.G @ (y - y_hat_m)
        self.P = self.P - self.G @ P_xy.T
