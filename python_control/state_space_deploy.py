"""
File: state_space_deploy.py

This module provides functionality to generate C++ header files representing discrete state-space systems defined using the Python `control` library. It automates the conversion of Python-based state-space models (A, B, C, D matrices and sampling time) into C++ code, facilitating deployment in C++ projects.
"""
import os
import sys
sys.path.append(os.getcwd())

import control
import inspect

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from python_control.control_deploy import ControlDeploy


class StateSpaceDeploy:
    """
    A class to generate C++ code for discrete state-space systems.
    This class provides a static method to convert a `control.StateSpace` object
    into a C++ header file containing the necessary definitions and functions
    to represent the state-space model in C++.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_state_space_cpp_code(
            state_space: control.StateSpace, file_name=None, number_of_delay=0):
        """
        Generates C++ code for a discrete state-space system.
        Args:
            state_space (control.StateSpace): The state-space system to convert.
            file_name (str, optional): The base name for the generated C++ file.
                If not provided, the caller's file name will be used.
            number_of_delay (int, optional): The number of delays in the system.
                Defaults to 0.
        Returns:
            list: A list of file names of the generated C++ files.
        Raises:
            ValueError: If the input state_space is not a valid control.StateSpace object.
        """
        deployed_file_names = []

        ControlDeploy.restrict_data_type(state_space.A.dtype.name)

        type_name = NumpyDeploy.check_dtype(state_space.A)

        # %% inspect arguments
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
        # Get the caller's file name
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = file_name

        # %% code generation
        code_file_name = caller_file_name_without_ext + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create A, B, C, D matrices
        exec(f"{variable_name}_A = state_space.A")
        A_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_A, caller_file_name_without_ext)")
        exec(f"{variable_name}_B = state_space.B")
        B_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_B, caller_file_name_without_ext)")
        exec(f"{variable_name}_C = state_space.C")
        C_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_C, caller_file_name_without_ext)")
        exec(f"{variable_name}_D = state_space.D")
        D_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_D, caller_file_name_without_ext)")

        deployed_file_names.append(A_file_name)
        deployed_file_names.append(B_file_name)
        deployed_file_names.append(C_file_name)
        deployed_file_names.append(D_file_name)

        A_file_name_no_extension = A_file_name.split(".")[0]
        B_file_name_no_extension = B_file_name.split(".")[0]
        C_file_name_no_extension = C_file_name.split(".")[0]
        D_file_name_no_extension = D_file_name.split(".")[0]

        # create state-space cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{A_file_name}\"\n"
        code_text += f"#include \"{B_file_name}\"\n"
        code_text += f"#include \"{C_file_name}\"\n"
        code_text += f"#include \"{D_file_name}\"\n\n"
        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonControl;\n\n"

        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {number_of_delay};\n\n"

        code_text += f"using A_Type = {A_file_name_no_extension}::type;\n\n"

        code_text += f"using B_Type = {B_file_name_no_extension}::type;\n\n"

        code_text += f"using C_Type = {C_file_name_no_extension}::type;\n\n"

        code_text += f"using D_Type = {D_file_name_no_extension}::type;\n\n"

        code_text += "constexpr std::size_t INPUT_SIZE = B_Type::ROWS;\n"
        code_text += "constexpr std::size_t STATE_SIZE = A_Type::COLS;\n"
        code_text += "constexpr std::size_t OUTPUT_SIZE = C_Type::COLS;\n\n"

        code_text += "using type = DiscreteStateSpace_Type<\n" + \
            "    A_Type, B_Type, C_Type, D_Type, NUMBER_OF_DELAY>;\n\n"

        code_text += "inline auto make(void) -> type {\n\n"

        code_text += f"  {type_name} dt = static_cast<{type_name}>({state_space.dt});\n\n"

        code_text += f"  auto A = {A_file_name_no_extension}::make();\n\n"

        code_text += f"  auto B = {B_file_name_no_extension}::make();\n\n"

        code_text += f"  auto C = {C_file_name_no_extension}::make();\n\n"

        code_text += f"  auto D = {D_file_name_no_extension}::make();\n\n"

        code_text += f"  return make_DiscreteStateSpace<NUMBER_OF_DELAY>(\n" + \
            "    A, B, C, D, dt);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
