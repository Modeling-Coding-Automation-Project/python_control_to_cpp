import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import matplotlib.pyplot as plt
import sympy
from sympy import symbols, lambdify

from python_control.kalman_filter import ExtendedKalmanFilter
from python_control.kalman_filter_deploy import KalmanFilterDeploy


def create_function_code_from_sympy(sym_object, file_name):
    with open("function_" + file_name + ".py", 'w') as f:
        code_text = ""
        code_text += f"import numpy as np\nfrom math import *\n\ndef fxu_func("

        fxu_symbols = fxu.free_symbols
        for i, symbol in enumerate(fxu_symbols):
            code_text += f"{symbol}"
            if i == len(fxu_symbols) - 1:
                code_text += "):\n\n"
            else:
                code_text += ", "

        code_text += f"    return np.array({fxu.tolist()})\n"

        f.write(code_text)

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

time = symbols('time')
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

# Save functions to separate files
code = KalmanFilterDeploy.create_function_code_from_sympy(fxu, "fxu")
KalmanFilterDeploy.write_code_to_file(code, "function_fxu.py")
