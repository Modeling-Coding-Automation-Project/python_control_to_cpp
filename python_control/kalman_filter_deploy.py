import os
import sys
sys.path.append(os.getcwd())

import inspect
import ast
import astor
import numpy as np

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from python_control.control_deploy import ControlDeploy


class KalmanFilterDeploy:
    def __init__(self):
        pass

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

        calculation_code = f"{sym_object.tolist()}"

        code_text += f"    return np.array({calculation_code})\n\n\n"

        return code_text, arguments_text

    @staticmethod
    def create_interface_code(sym_object, arguments_text, X, U=None):
        sym_symbols = sym_object.free_symbols

        code_text = ""
        code_text += "def function(X"

        if U is not None:
            code_text += ", U, Parameters=None):\n\n"
        else:
            code_text += ", Parameters=None):\n\n"

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

        for symbol in sym_symbols:
            code_text += f"    {symbol} = "
            code_text += f"Parameters.{symbol}\n"

        code_text += "\n"

        code_text += "    return sympy_function("
        code_text += arguments_text
        code_text += ")\n"

        return code_text

    @staticmethod
    def write_function_code_from_sympy(sym_object, sym_object_name, X, U=None):
        header_code = "import numpy as np\nfrom math import *\n\n\n"

        sympy_function_code, arguments_text = KalmanFilterDeploy.create_sympy_code(
            sym_object)

        interface_code = KalmanFilterDeploy.create_interface_code(
            sym_object, arguments_text, X, U)

        total_code = header_code + sympy_function_code + interface_code

        ControlDeploy.write_to_file(
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

    @staticmethod
    def generate_LKF_cpp_code(lkf):
        deployed_file_names = []

        ControlDeploy.restrict_data_type(lkf.A.dtype.name)

        type_name = NumpyDeploy.check_dtype(lkf.A)

        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is lkf:
                variable_name = name
                break

        code_file_name = "python_control_gen_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create A, B, C, D matrices
        exec(f"{variable_name}_A = lkf.A")
        A_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_A)")
        exec(f"{variable_name}_B = lkf.B")
        B_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_B)")
        exec(f"{variable_name}_C = lkf.C")
        C_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_C)")

        # D is empty
        D = np.zeros((lkf.C.shape[0], lkf.B.shape[1]))
        exec(f"{variable_name}_D = D")
        D_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_D)")

        deployed_file_names.append(A_file_name)
        deployed_file_names.append(B_file_name)
        deployed_file_names.append(C_file_name)
        deployed_file_names.append(D_file_name)

        A_file_name_no_extension = A_file_name.split(".")[0]
        B_file_name_no_extension = B_file_name.split(".")[0]
        C_file_name_no_extension = C_file_name.split(".")[0]
        D_file_name_no_extension = D_file_name.split(".")[0]


class PowerReplacer(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and node.right.value == 2:

            return ast.BinOp(left=node.left, op=ast.Mult(), right=node.left)
        return node

    def transform_code(self, source_code):
        tree = ast.parse(source_code)
        transformer = ast.NodeTransformer()
        transformed_tree = transformer.visit(tree)
        return astor.to_source(transformed_tree)
