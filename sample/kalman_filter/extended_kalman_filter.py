import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import symbols

from python_control.kalman_filter import ExtendedKalmanFilter


# %% bicycle model example
# state x: [x, y, theta]
# control u: [v, alpha]
# observation y: [r_p, angle_p]

alpha = symbols('alpha')
x = symbols('x')
y = symbols('y')
v = symbols('v')
wheelbase = symbols('wheelbase')
radius = symbols('radius')
theta = symbols('theta')
p_x = symbols('p_x')
p_y = symbols('p_y')

time = symbols('t')
d = v * time
beta = (d / wheelbase) * sympy.tan(alpha)
r = wheelbase / sympy.tan(alpha)

fxu = sympy.Matrix([[x - r * sympy.sin(theta) + r * sympy.sin(theta + beta)],
                    [y + r * sympy.cos(theta) - r * sympy.cos(theta + beta)],
                    [theta + beta]])
fxu_jacobian = fxu.jacobian(sympy.Matrix([x, y, theta]))
print("fxu_jacobian:\n", fxu_jacobian)

hx = sympy.Matrix([[sympy.sqrt((p_x - x)**2 + (p_y - y)**2)],
                   [sympy.atan2(p_y - y, p_x - x) - theta]])
hx_jacobian = hx.jacobian(sympy.Matrix([x, y, theta]))
print("hx_jacobian:\n", hx_jacobian)
