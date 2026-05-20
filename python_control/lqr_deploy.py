"""
File: lqr_deploy.py

This module provides utilities for generating C++ header files
that implement LQR (Linear Quadratic Regulator) and
LQI (Linear Quadratic Integrator) controllers from Python-based
system matrices and weighting matrices.
The generated C++ code is intended for use with a compatible C++ control library,
enabling seamless deployment of control algorithms designed in Python to C++ environments.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import copy
import numpy as np
import control
import inspect

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy


class LQR_METHOD:
    ARIMOTO_POTTER = 0
    DARE = 1


class LQR_Deploy:
    """
    A class for generating C++ code for Linear Quadratic Regulator (LQR) controllers.
    This class provides a static method to generate C++ header files that define the LQR controller
    based on the provided state-space matrices and weighting matrices.
    Attributes:
        None
    Methods:
        generate_LQR_cpp_code(A, B, Q, R, file_name=None):
            Generates C++ code for an LQR controller using the provided state-space matrices and weighting matrices.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_LQR_cpp_code(
            A, B, Q, R, file_name=None, method=LQR_METHOD.ARIMOTO_POTTER):
        """
        Generates C++ code for an LQR controller using the provided state-space matrices and weighting matrices.
        Args:
            A (np.ndarray): State matrix of the system.
            B (np.ndarray): Input matrix of the system.
            Q (np.ndarray): State weighting matrix for the LQR controller.
            R (np.ndarray): Input weighting matrix for the LQR controller.
            file_name (str, optional): The base name for the generated C++ header file. If None, the caller's file name is used.
            method (int, optional): The method to use for LQR computation. Defaults to LQR_METHOD.ARIMOTO_POTTER.
        Returns:
            list: A list of file names for the generated C++ header files.
        Raises:
            ValueError: If the input matrices are not compatible or if the data type is unsupported.
        """
        deployed_file_names = []

        ControlDeploy.restrict_data_type(A.dtype.name)

        type_name = NumpyDeploy.check_dtype(A)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's file name
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = file_name

        # %% code generation
        variable_name = "LQR"

        code_file_name = caller_file_name_without_ext + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create A, B matrices
        # Use locals_map + eval (avoids exec and matches pattern in state_space_deploy.py)
        locals_map = {
            f"{variable_name}_A": copy.deepcopy(A),
            "caller_file_name_without_ext": caller_file_name_without_ext,
        }
        A_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_A, " +
            "file_name=caller_file_name_without_ext)", globals(), locals_map)

        locals_map = {
            f"{variable_name}_B": copy.deepcopy(B),
            "caller_file_name_without_ext": caller_file_name_without_ext,
        }
        B_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_B, " +
            "file_name=caller_file_name_without_ext)", globals(), locals_map)

        deployed_file_names.append(A_file_name)
        deployed_file_names.append(B_file_name)

        A_file_name_no_extension = A_file_name.split(".")[0]
        B_file_name_no_extension = B_file_name.split(".")[0]

        # create state-space cpp code
        code_text = ""

        file_header_macro_name = code_file_name.upper() + "_HPP_"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{A_file_name}\"\n"
        code_text += f"#include \"{B_file_name}\"\n\n"
        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n\n"

        if LQR_METHOD.ARIMOTO_POTTER == method:
            code_text += "constexpr std::size_t LQR_METHOD = LQR_METHOD_ARIMOTO_POTTER;\n\n"
        elif LQR_METHOD.DARE == method:
            code_text += "constexpr std::size_t LQR_METHOD = LQR_METHOD_DARE;\n\n"
        else:
            raise ValueError("Invalid LQR method specified.")

        code_text += f"using A_Type = {A_file_name_no_extension}::type;\n\n"

        code_text += f"using B_Type = {B_file_name_no_extension}::type;\n\n"

        code_text += "constexpr std::size_t STATE_SIZE = A_Type::ROWS;\n"
        code_text += "constexpr std::size_t INPUT_SIZE = B_Type::COLS;\n\n"

        code_text += f"using Q_Type = DiagMatrix_Type<{type_name}, STATE_SIZE>;\n\n"

        code_text += f"using R_Type = DiagMatrix_Type<{type_name}, INPUT_SIZE>;\n\n"

        code_text += "using type = LQR_Type<" + \
            "A_Type, B_Type, Q_Type, R_Type, LQR_METHOD>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += f"  auto A = {A_file_name_no_extension}::make();\n\n"

        code_text += f"  auto B = {B_file_name_no_extension}::make();\n\n"

        code_text += "  auto Q = make_DiagMatrix<STATE_SIZE>(\n"
        for i in range(Q.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(Q[i, i]) + ")"
            if i == Q.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += "  );\n\n"

        code_text += "  auto R = make_DiagMatrix<INPUT_SIZE>(\n"
        for i in range(R.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(R[i, i]) + ")"
            if i == R.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += "  );\n\n"

        code_text += "  return make_LQR<LQR_METHOD>(A, B, Q, R);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names


class LQI_Deploy:
    """
    A class for generating C++ code for Linear Quadratic Integrator (LQI) controllers.
    This class provides a static method to generate C++ header files that define the LQI controller
    based on the provided state-space matrices and weighting matrices.
    Attributes:
        None
    Methods:
        generate_LQI_cpp_code(A, B, C, Q_ex, R_ex, file_name=None):
            Generates C++ code for an LQI controller using the provided state-space matrices and weighting matrices.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_LQI_cpp_code(
            A, B, C, Q_ex, R_ex, file_name=None, method=LQR_METHOD.ARIMOTO_POTTER):
        """
        Generates C++ code for an LQI controller using the provided state-space matrices and weighting matrices.
        Args:
            A (np.ndarray): State matrix of the system.
            B (np.ndarray): Input matrix of the system.
            C (np.ndarray): Output matrix of the system.
            Q_ex (np.ndarray): Extended state weighting matrix for the LQI controller.
            R_ex (np.ndarray): Input weighting matrix for the LQI controller.
            file_name (str, optional): The base name for the generated C++ header file. If None, the caller's file name is used.
            method (int, optional): The method to use for LQI computation. Defaults to LQR_METHOD.ARIMOTO_POTTER.
        Returns:
            list: A list of file names for the generated C++ header files.
        Raises:
            ValueError: If the input matrices are not compatible or if the data type is unsupported.
        """
        deployed_file_names = []

        ControlDeploy.restrict_data_type(A.dtype.name)

        type_name = NumpyDeploy.check_dtype(A)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's file name
        if file_name is None:
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = file_name

        # %% code generation
        variable_name = "LQI"

        code_file_name = caller_file_name_without_ext + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create A, B, C matrices (use locals_map to avoid exec)
        locals_map = {
            f"{variable_name}_A": copy.deepcopy(A),
            "caller_file_name_without_ext": caller_file_name_without_ext,
        }
        A_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_A, " +
            "file_name=caller_file_name_without_ext)", globals(), locals_map)

        locals_map = {
            f"{variable_name}_B": copy.deepcopy(B),
            "caller_file_name_without_ext": caller_file_name_without_ext,
        }
        B_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_B, " +
            "file_name=caller_file_name_without_ext)", globals(), locals_map)

        locals_map = {
            f"{variable_name}_C": copy.deepcopy(C),
            "caller_file_name_without_ext": caller_file_name_without_ext,
        }
        C_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_C, " +
            "file_name=caller_file_name_without_ext)", globals(), locals_map)

        deployed_file_names.append(A_file_name)
        deployed_file_names.append(B_file_name)
        deployed_file_names.append(C_file_name)

        A_file_name_no_extension = A_file_name.split(".")[0]
        B_file_name_no_extension = B_file_name.split(".")[0]
        C_file_name_no_extension = C_file_name.split(".")[0]

        # create state-space cpp code
        code_text = ""

        file_header_macro_name = code_file_name.upper() + "_HPP_"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{A_file_name}\"\n"
        code_text += f"#include \"{B_file_name}\"\n"
        code_text += f"#include \"{C_file_name}\"\n\n"

        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n\n"

        if method == LQR_METHOD.ARIMOTO_POTTER:
            code_text += "constexpr std::size_t LQR_METHOD = LQR_METHOD_ARIMOTO_POTTER;\n\n"
        elif method == LQR_METHOD.DARE:
            code_text += "constexpr std::size_t LQR_METHOD = LQR_METHOD_DARE;\n\n"
        else:
            raise ValueError("Invalid LQR method specified.")

        code_text += f"auto A = {A_file_name_no_extension}::make();\n\n"

        code_text += f"auto B = {B_file_name_no_extension}::make();\n\n"

        code_text += f"auto C = {C_file_name_no_extension}::make();\n\n"

        code_text += "constexpr std::size_t STATE_SIZE = decltype(A)::ROWS;\n"
        code_text += "constexpr std::size_t INPUT_SIZE = decltype(B)::COLS;\n"
        code_text += "constexpr std::size_t OUTPUT_SIZE = decltype(C)::ROWS;\n"
        code_text += "constexpr std::size_t Q_EX_SIZE = STATE_SIZE + OUTPUT_SIZE;\n\n"

        code_text += "auto Q_ex = make_DiagMatrix<Q_EX_SIZE>(\n"
        for i in range(Q_ex.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(Q_ex[i, i]) + ")"
            if i == Q_ex.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += ");\n\n"

        code_text += "auto R_ex = make_DiagMatrix<INPUT_SIZE>(\n"
        for i in range(R_ex.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(R_ex[i, i]) + ")"
            if i == R_ex.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += ");\n\n"

        code_text += "using type = LQI_Type<" + \
            "decltype(A), decltype(B), decltype(C), decltype(Q_ex), decltype(R_ex), LQR_METHOD>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += "    return make_LQI<LQR_METHOD>(A, B, C, Q_ex, R_ex);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
