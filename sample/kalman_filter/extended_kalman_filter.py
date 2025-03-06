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
p_x = symbols('p_x')
p_y = symbols('p_y')

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

hx = sympy.Matrix([[sympy.sqrt((p_x - x) ** 2 + (p_y - y) ** 2)],
                   [sympy.atan2(p_y - y, p_x - x) - theta]])
hx_jacobian = hx.jacobian(X)
print("hx_jacobian:\n", hx_jacobian)

# Save functions to separate files
KalmanFilterDeploy.write_state_function_code_from_sympy(fxu, X, U)
KalmanFilterDeploy.write_state_function_code_from_sympy(fxu_jacobian, X, U)

KalmanFilterDeploy.write_measurement_function_code_from_sympy(hx, X)
KalmanFilterDeploy.write_measurement_function_code_from_sympy(hx_jacobian, X)

# %% design EKF

landmark = np.array([[0.0], [0.0]])


class Parameters:
    def __init__(self, delta_time, wheelbase, p_x, p_y):
        self.delta_time = delta_time
        self.wheelbase = wheelbase
        self.p_x = p_x
        self.p_y = p_y


Parameters_ekf = Parameters(
    delta_time=0.1, wheelbase=0.5, p_x=landmark[0, 0], p_y=landmark[1, 0])

Q_ekf = np.diag([0.1, 0.1, 0.1])
R_ekf = np.diag([0.1, 0.1])

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
simulation_time = 20.0


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

    x_next = x + dx

    x_dif = (landmark[0, 0] - x_next[0, 0])
    y_dif = (landmark[1, 0] - x_next[1, 0])
    theta = x_next[2, 0]
    r = math.sqrt(x_dif * x_dif + y_dif * y_dif)
    phi = math.atan2(y_dif, x_dif) - theta
    y_next = np.array([[r], [phi]])

    return x_next, y_next


def run_simulation():

    x = np.array([[2.0], [6.0], [0.3]])  # initial x, y, theta
    u = np.array([[1.1], [0.1]])  # inputs v, steering_angle

    x_true = []
    y_measured = []
    time = np.arange(0, simulation_time, sim_delta_time)
    for i in range(round(simulation_time / sim_delta_time)):
        x, y = move(x, u, sim_delta_time,
                    sim_wheelbase)  # simulate robot

        x_true.append(x)
        y_measured.append(y)

    x_true = np.array(x_true)
    y_measured = np.array(y_measured)

    # plot
    fig, axs = plt.subplots(3, 2)
    fig.suptitle("EKF for bicycle model results")

    axs[0, 0].plot(time, x_true[:, 0, 0])
    axs[0, 0].set_ylabel('x position')
    axs[0, 0].grid(True)

    axs[1, 0].plot(time, x_true[:, 1, 0])
    axs[1, 0].set_ylabel('y position')
    axs[1, 0].grid(True)

    axs[2, 0].plot(time, x_true[:, 2, 0])
    axs[2, 0].set_ylabel('theta')
    axs[2, 0].set_xlabel('time')
    axs[2, 0].grid(True)

    plt.tight_layout()


run_simulation()

plt.show()
