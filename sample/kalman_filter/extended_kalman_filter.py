"""
Extended Kalman Filter sample code

Reference URL:
https://inzkyk.xyz/kalman_filter/extended_kalman_filters/#subsection:11.4.1
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import symbols

from python_control.kalman_filter import ExtendedKalmanFilter
from python_control.kalman_filter_deploy import KalmanFilterDeploy

# %% bicycle model example
# state X: [x, y, theta]
# input U: [v, alpha]
# output Y: [r_p, angle_p]

delta_time = symbols('delta_time')
alpha = symbols('alpha')
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
U = sympy.Matrix([[v], [alpha]])

d = v * delta_time
beta = (d / wheelbase) * sympy.tan(alpha)
r = wheelbase / sympy.tan(alpha)

fxu = sympy.Matrix([[x - r * sympy.sin(theta) + r * sympy.sin(theta + beta)],
                    [y + r * sympy.cos(theta) - r * sympy.cos(theta + beta)],
                    [theta + beta]])
fxu_jacobian = fxu.jacobian(X)
print("fxu_jacobian:\n", fxu_jacobian)

hx = sympy.Matrix([[sympy.sqrt((p_x - x)**2 + (p_y - y)**2)],
                   [sympy.atan2(p_y - y, p_x - x) - theta]])
hx_jacobian = hx.jacobian(X)
print("hx_jacobian:\n", hx_jacobian)

# Save functions to separate files
KalmanFilterDeploy.write_state_function_code_from_sympy(fxu, X, U)
KalmanFilterDeploy.write_state_function_code_from_sympy(fxu_jacobian, X, U)

KalmanFilterDeploy.write_measurement_function_code_from_sympy(hx, X)
KalmanFilterDeploy.write_measurement_function_code_from_sympy(hx_jacobian, X)
