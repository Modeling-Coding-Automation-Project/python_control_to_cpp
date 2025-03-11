"""
Extended Kalman Filter sample code

Reference URL:
https://inzkyk.xyz/kalman_filter/extended_kalman_filters/#subsection:11.4.1
"""
import os
import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import symbols

from python_control.kalman_filter import ExtendedKalmanFilter
from python_control.kalman_filter_deploy import KalmanFilterDeploy

# %% bicycle model example
# state X: [x, y, theta]
# input U: [v, steering_angle]
# output Y: [r_p, angle_p]

delta_time = symbols('delta_time')
steering_angle = symbols('steering_angle')
x = symbols('x')
y = symbols('y')
v = symbols('v')
wheelbase = symbols('wheelbase')
radius = symbols('radius')
theta = symbols('theta')
landmark_1_x = symbols('landmark_1_x')
landmark_1_y = symbols('landmark_1_y')
landmark_2_x = symbols('landmark_2_x')
landmark_2_y = symbols('landmark_2_y')

# define state X, input U
X = sympy.Matrix([[x], [y], [theta]])
U = sympy.Matrix([[v], [steering_angle]])

d = v * delta_time
beta = (d / wheelbase) * sympy.tan(steering_angle)
r = wheelbase / sympy.tan(steering_angle)

fxu = sympy.Matrix([[x - r * sympy.sin(theta) + r * sympy.sin(theta + beta)],
                    [y + r * sympy.cos(theta) - r * sympy.cos(theta + beta)],
                    [theta + beta]])
fxu_jacobian = fxu.jacobian(X)
print("fxu_jacobian:\n", fxu_jacobian)

hx = sympy.Matrix([[sympy.sqrt((landmark_1_x - x) ** 2 + (landmark_1_y - y) ** 2)],
                   [sympy.atan2(landmark_1_y - y, landmark_1_x - x) - theta],
                   [sympy.sqrt((landmark_2_x - x) ** 2 +
                               (landmark_2_y - y) ** 2)],
                   [sympy.atan2(landmark_2_y - y, landmark_2_x - x) - theta]])
hx_jacobian = hx.jacobian(X)
print("hx_jacobian:\n", hx_jacobian)

# Save functions to separate files
KalmanFilterDeploy.write_state_function_code_from_sympy(fxu, X, U)
KalmanFilterDeploy.write_state_function_code_from_sympy(fxu_jacobian, X, U)

KalmanFilterDeploy.write_measurement_function_code_from_sympy(hx, X)
KalmanFilterDeploy.write_measurement_function_code_from_sympy(hx_jacobian, X)

# %% design EKF

landmarks = np.array([[0.0, 10.0], [0.0, 10.0]])


class Parameters:
    def __init__(self, delta_time, wheelbase,
                 landmark_1_x, landmark_1_y, landmark_2_x, landmark_2_y):
        self.delta_time = delta_time
        self.wheelbase = wheelbase
        self.landmark_1_x = landmark_1_x
        self.landmark_1_y = landmark_1_y
        self.landmark_2_x = landmark_2_x
        self.landmark_2_y = landmark_2_y


Parameters_ekf = Parameters(
    delta_time=0.1, wheelbase=0.5,
    landmark_1_x=landmarks[0, 0], landmark_1_y=landmarks[1, 0],
    landmark_2_x=landmarks[0, 1], landmark_2_y=landmarks[1, 1])

Q_ekf = np.diag([1.0, 1.0, 1.0])
R_ekf = np.diag([10.0, 10.0, 10.0, 10.0])

import fxu
import fxu_jacobian
import hx
import hx_jacobian
ekf = ExtendedKalmanFilter(fxu.function, hx.function,
                           fxu_jacobian.function, hx_jacobian.function,
                           Q_ekf, R_ekf, Parameters_ekf)


# %% bicycle model simulation

sim_delta_time = 0.1
sim_wheelbase = Parameters_ekf.wheelbase
simulation_time = 50.0


def move(x, u, dt, wheelbase):
    hdg = x[2, 0]
    vel = u[0, 0]
    steering_angle = u[1, 0]
    dist = vel * dt

    if abs(steering_angle) > 0.001:  # is robot rataing?
        beta = (dist / wheelbase) * math.tan(steering_angle)
        r = wheelbase / math.tan(steering_angle)  # radius

        dx = np.array([[-r * math.sin(hdg) + r * math.sin(hdg + beta)],
                       [r * math.cos(hdg) - r * math.cos(hdg + beta)],
                       [beta]])
    else:  # moving in straight line
        dx = np.array([[dist * math.cos(hdg)],
                       [dist * math.sin(hdg)],
                       [0]])

    x_next = x + dx + np.array([[np.random.randn() * 0.0],
                                [np.random.randn() * 0.0],
                                [np.random.randn() * 0.0]])

    theta = x_next[2, 0]
    # landmark 1
    x_dif = (landmarks[0, 0] - x_next[0, 0])
    y_dif = (landmarks[1, 0] - x_next[1, 0])
    r_1 = math.sqrt(x_dif * x_dif + y_dif * y_dif) + np.random.randn() * 0.0
    phi_1 = math.atan2(y_dif, x_dif) - theta + np.random.randn() * 0.0

    # landmark 2
    x_dif = (landmarks[0, 1] - x_next[0, 0])
    y_dif = (landmarks[1, 1] - x_next[1, 0])
    r_2 = math.sqrt(x_dif * x_dif + y_dif * y_dif) + np.random.randn() * 0.0
    phi_2 = math.atan2(y_dif, x_dif) - theta + np.random.randn() * 0.0

    y_next = np.array([[r_1], [phi_1], [r_2], [phi_2]])

    return x_next, y_next


def run_simulation():

    x = np.array([[2.0], [6.0], [0.3]])  # initial x, y, theta
    u = np.array([[1.1], [0.1]])  # inputs v, steering_angle

    ekf.x_hat = np.array([[0.0], [0.0], [0.0]])  # initial state

    x_true = []
    x_estimated = []
    y_measured = []
    time = np.arange(0, simulation_time, sim_delta_time)

    u_true = []

    for i in range(round(simulation_time / sim_delta_time)):
        x, y = move(x, u, sim_delta_time,
                    sim_wheelbase)  # simulate robot

        x_true.append(x)
        u_true.append(u)
        y_measured.append(y)

        # estimate
        ekf.predict(u)
        ekf.update(y)

        x_estimated.append(ekf.x_hat)

    x_true = np.array(x_true)
    u_true = np.array(u_true)
    y_measured = np.array(y_measured)
    x_estimated = np.array(x_estimated)

    # plot
    fig, axs = plt.subplots(3, 2)
    fig.suptitle("EKF for bicycle model results")

    axs[0, 0].plot(time, x_true[:, 0, 0], label='true')
    axs[0, 0].plot(time, x_estimated[:, 0, 0], label='estimated')
    axs[0, 0].legend()
    axs[0, 0].set_ylabel('x position (state)')
    axs[0, 0].grid(True)

    axs[1, 0].plot(time, x_true[:, 1, 0], label='true')
    axs[1, 0].plot(time, x_estimated[:, 1, 0], label='estimated')
    axs[1, 0].legend()
    axs[1, 0].set_ylabel('y position (state)')
    axs[1, 0].grid(True)

    axs[2, 0].plot(time, x_true[:, 2, 0], label='true')
    axs[2, 0].plot(time, x_estimated[:, 2, 0], label='estimated')
    axs[2, 0].legend()
    axs[2, 0].set_ylabel('theta (state)')
    axs[2, 0].set_xlabel('time')
    axs[2, 0].grid(True)

    axs[0, 1].plot(time, u_true[:, 0, 0], label='v')
    axs[0, 1].plot(time, u_true[:, 1, 0], label='steering angle')
    axs[0, 1].legend()
    axs[0, 1].set_ylabel('inputs')
    axs[0, 1].grid(True)

    axs[1, 1].plot(time, y_measured[:, 0, 0], label='landmark 1')
    axs[1, 1].plot(time, y_measured[:, 2, 0], label='landmark 2')
    axs[1, 1].legend()
    axs[1, 1].set_ylabel('r (output)')
    axs[1, 1].grid(True)

    axs[2, 1].plot(time, y_measured[:, 1, 0], label='landmark 1')
    axs[2, 1].plot(time, y_measured[:, 3, 0], label='landmark 2')
    axs[2, 1].legend()
    axs[2, 1].set_ylabel('phi (output)')
    axs[2, 1].grid(True)


run_simulation()

plt.show()
