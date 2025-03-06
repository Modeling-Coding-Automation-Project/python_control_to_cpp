import os
import sys
sys.path.append(os.getcwd())

import inspect

from python_control.kalman_filter import LinearKalmanFilter
from python_control.kalman_filter import ExtendedKalmanFilter


class KalmanFilterDeploy:
    @staticmethod
    def create_sympy_code(sym_object):
        code_text = ""
        code_text += "def sympy_function("

        arguments_text = ""

        sym_symbols = sym_object.free_symbols
        for i, symbol in enumerate(sym_symbols):
            arguments_text += f"{symbol}"
            if i == len(sym_symbols) - 1:
                break
            else:
                arguments_text += ", "

        code_text += arguments_text + "):\n\n"

        code_text += f"    return np.array({sym_object.tolist()})\n\n\n"

        return code_text, arguments_text

    @staticmethod
    def create_interface_code(sym_object, arguments_text, X, U=None):
        sym_symbols = sym_object.free_symbols

        code_text = ""
        code_text += "def function(X"

        if U is not None:
            code_text += ", U):\n\n"
        else:
            code_text += "):\n\n"

        for i in range(X.shape[0]):
            if X[i] in sym_symbols:
                code_text += f"    {X[i]} = X[{i}, 0]\n"
                sym_symbols.remove(X[i])

        if U is not None:
            for i in range(U.shape[0]):
                if U[i] in sym_symbols:
                    code_text += f"    {U[i]} = U[{i}, 0]\n"
                    sym_symbols.remove(U[i])

        code_text += "\n"
        code_text += "    # You need to set appropriate parameters in below:\n"

        for symbol in sym_symbols:
            code_text += f"    {symbol} = 0.0\n"

        code_text += "\n"

        code_text += "    return sympy_function("
        code_text += arguments_text
        code_text += ")\n"

        return code_text

    @staticmethod
    def write_code_to_file(code_text, file_name):
        with open(file_name, 'w') as f:
            f.write(code_text)

    @staticmethod
    def write_function_code_from_sympy(sym_object, sym_object_name, X, U=None):
        header_code = "import numpy as np\nfrom math import *\n\n\n"

        sympy_function_code, arguments_text = \
            KalmanFilterDeploy.create_sympy_code(sym_object)

        interface_code = KalmanFilterDeploy.create_interface_code(
            sym_object, arguments_text, X, U)

        total_code = header_code + sympy_function_code + interface_code

        KalmanFilterDeploy.write_code_to_file(
            total_code, f"{sym_object_name}.py")

    @staticmethod
    def write_state_function_code_from_sympy(sym_object, X, U=None):
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        sym_object_name = None
        for name, value in caller_locals.items():
            if value is sym_object:
                sym_object_name = name
                break

        KalmanFilterDeploy.write_function_code_from_sympy(
            sym_object, sym_object_name, X, U)

    @staticmethod
    def write_measurement_function_code_from_sympy(sym_object, X):
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        sym_object_name = None
        for name, value in caller_locals.items():
            if value is sym_object:
                sym_object_name = name
                break

        KalmanFilterDeploy.write_function_code_from_sympy(
            sym_object, sym_object_name, X, U=None)
