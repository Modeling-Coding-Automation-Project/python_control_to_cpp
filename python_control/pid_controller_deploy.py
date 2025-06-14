"""
This module provides functionality to generate C++ header files for deploying discrete PID controllers
defined in Python. It automates the translation of Python-based PID controller parameters and configuration
into C++ code, facilitating integration with C++ projects.
"""
import os
import sys
sys.path.append(os.getcwd())

import inspect
import numpy as np

from python_control.pid_controller import DiscretePID_Controller
from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from python_control.control_deploy import ControlDeploy


class DiscretePID_ControllerDeploy:
    """
    A class for generating C++ code for Discrete PID Controllers.
    This class provides static methods to generate C++ header files based on Python DiscretePID_Controller objects.
    The generated code includes type definitions, constants, and functions to create instances of the DiscretePID_Controller.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_PID_cpp_code(pid: DiscretePID_Controller, file_name=None):
        """
        Generates C++ code for a Discrete PID Controller based on the provided Python DiscretePID_Controller object.
        Args:
            pid: A DiscretePID_Controller object containing the PID parameters.
            file_name: Optional; the base name for the generated C++ header file.
        Returns:
            A list of file names for the generated C++ header files.
        """
        deployed_file_names = []

        ControlDeploy.restrict_data_type(pid.data_type)

        type_name = NumpyDeploy.check_dtype(np.array([[pid.Kp]]))

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is pid:
                variable_name = name
                break
        # Get the caller's file name
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = file_name

        delta_time = pid.delta_time
        Kp = pid.Kp
        Ki = pid.Ki
        Kd = pid.Kd
        N = pid.N
        Kb = pid.Kb

        minimum_output = pid.minimum_output
        if np.isinf(minimum_output):
            minimum_output = -1.0e10

        maximum_output = pid.maximum_output
        if np.isinf(maximum_output):
            maximum_output = 1.0e10

        # %% code generation
        code_file_name = caller_file_name_without_ext + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create state-space cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonControl;\n\n"

        code_text += f"constexpr {type_name} DELTA_TIME = {delta_time};\n"
        code_text += f"constexpr {type_name} Kp = {Kp};\n"
        code_text += f"constexpr {type_name} Ki = {Ki};\n"
        code_text += f"constexpr {type_name} Kd = {Kd};\n"
        code_text += f"constexpr {type_name} N = {N};\n"
        code_text += f"constexpr {type_name} Kb = {Kb};\n"
        code_text += f"constexpr {type_name} MINIMUM_OUTPUT = {minimum_output};\n"
        code_text += f"constexpr {type_name} MAXIMUM_OUTPUT = {maximum_output};\n\n"

        code_text += f"using type = DiscretePID_Controller_Type<{type_name}>;\n\n"

        code_text += "inline auto make(void) -> type {\n\n"

        code_text += f"  return make_DiscretePID_Controller<{type_name}>(\n" + \
            "    DELTA_TIME,\n" + \
            "    Kp,\n" + \
            "    Ki,\n" + \
            "    Kd,\n" + \
            "    N,\n" + \
            "    Kb,\n" + \
            "    MINIMUM_OUTPUT,\n" + \
            "    MAXIMUM_OUTPUT\n" + \
            "    );\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
