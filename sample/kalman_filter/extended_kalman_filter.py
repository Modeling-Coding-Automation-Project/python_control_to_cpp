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
    return x + dx


def run_simulation():

    sim_pos = np.array([[2.0], [6.0], [0.3]])  # x, y, 旋回角
    u = np.array([[1.1], [0.1]])  # 操縦コマンド (速度と旋回角)

    track = []
    time = np.arange(0, simulation_time, sim_delta_time)
    for i in range(round(simulation_time / sim_delta_time)):
        sim_pos = move(sim_pos, u, sim_delta_time,
                       sim_wheelbase)  # simulate robot
        track.append(sim_pos)

    track = np.array(track)

    # plot
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(time, track[:, 0, 0])
    axs[0].set_ylabel('x position')
    axs[0].grid()

    axs[1].plot(time, track[:, 1, 0])
    axs[1].set_ylabel('y position')
    axs[1].grid()

    axs[2].plot(time, track[:, 2, 0])
    axs[2].set_ylabel('theta')
    axs[2].set_xlabel('time')
    axs[2].grid()

    plt.tight_layout()


run_simulation()

plt.show()
