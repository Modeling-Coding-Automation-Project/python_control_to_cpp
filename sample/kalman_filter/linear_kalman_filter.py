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


# 例の使用方法
if __name__ == "__main__":
    # 状態遷移行列
    A = np.array([[1, 1], [0, 1]])
    # 制御行列
    B = np.array([[0.5], [1]])
    # 観測行列
    C = np.array([[1, 0]])
    # プロセスノイズ共分散行列
    Q = np.array([[1, 0], [0, 1]])
    # 観測ノイズ共分散行列
    R = np.array([[1]])

    # カルマンフィルタのインスタンスを作成
    kf = LinearKalmanFilter(A, B, C, Q, R)

    # 制御入力
    u = np.array([[0]])

    # 観測値
    z = np.array([[1]])

    # 予測ステップ
    kf.predict(u)
    print("予測後の状態:")
    print(kf.get_x_hat())

    # 更新ステップ
    kf.update(z)
    print("更新後の状態:")
    print(kf.get_x_hat())
