import os
import sys
sys.path.append(os.getcwd())

from python_control.kalman_filter import LinearKalmanFilter
from python_control.kalman_filter import ExtendedKalmanFilter


class KalmanFilterDeploy:
    @staticmethod
    def create_function_code_from_sympy(sym_object, file_name):
        code_text = ""
        code_text += f"import numpy as np\nfrom math import *\n\ndef fxu_func("

        sym_symbols = sym_object.free_symbols
        for i, symbol in enumerate(sym_symbols):
            code_text += f"{symbol}"
            if i == len(sym_symbols) - 1:
                code_text += "):\n\n"
            else:
                code_text += ", "

        code_text += f"    return np.array({sym_object.tolist()})\n"

        return code_text

    @staticmethod
    def write_code_to_file(code_text, file_name):
        with open(file_name, 'w') as f:
            f.write(code_text)
