import os
import sys
sys.path.append(os.getcwd())

import inspect

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy


class StateSpaceDeploy:
    def __init__(self):
        pass

    @staticmethod
    def check_data_type(state_space):
        if state_space.A.dtype.name == 'float64':
            return True
        elif state_space.A.dtype.name == 'float32':
            return True
        else:
            return False

    @staticmethod
    def generate_state_space_cpp_code(state_space):

        if not StateSpaceDeploy.check_data_type(state_space):
            raise ValueError(
                "Data type not supported. Please use float32 or float64")

        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is state_space:
                variable_name = name
                break

        # create A, B, C, D matrices
        exec(f"{variable_name}_A = state_space.A")
        A_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_A)")
        exec(f"{variable_name}_B = state_space.B")
        B_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_B)")
        exec(f"{variable_name}_C = state_space.C")
        C_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_C)")
        exec(f"{variable_name}_D = state_space.D")
        D_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_D)")

        # create state-space cpp code
        code_text = ""
        code_text += "#ifndef __PYTHON_CONTROL_GEN_" + variable_name.upper() + \
            "_HPP__\n"
        code_text += "#define __PYTHON_CONTROL_GEN_" + \
            variable_name.upper() + "_HPP__\n\n"

        code_text += "#include \"python_control.hpp\"\n\n"

        code_text += "namespace python_control_gen_" + variable_name + " {\n\n"

        code_text += "using namespace PythonControl;\n\n"

        code_text += "auto A = "

        pass
