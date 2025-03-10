import os
import sys
sys.path.append(os.getcwd())

import inspect

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from python_control.control_deploy import ControlDeploy


class TransferFunctionDeploy(ControlDeploy):
    def __init__(self):
        super().__init__()

    @staticmethod
    def generate_transfer_function_cpp_code(transfer_function):
        deployed_file_names = []

        ControlDeploy.restrict_data_type(transfer_function)

        type_name = NumpyDeploy.check_dtype(transfer_function.A)

        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is transfer_function:
                variable_name = name
                break

        code_file_name = "python_control_gen_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create A, B, C, D matrices
        exec(f"{variable_name}_A = transfer_function.A")
        A_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_A)")
        exec(f"{variable_name}_B = transfer_function.B")
        B_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_B)")
        exec(f"{variable_name}_C = transfer_function.C")
        C_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_C)")
        exec(f"{variable_name}_D = transfer_function.D")
        D_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_D)")

        A_file_name_no_extension = A_file_name.split(".")[0]
        B_file_name_no_extension = B_file_name.split(".")[0]
        C_file_name_no_extension = C_file_name.split(".")[0]
        D_file_name_no_extension = D_file_name.split(".")[0]

        # create state-space cpp code
        code_text = ""
        code_text += "#ifndef __PYTHON_CONTROL_GEN_" + variable_name.upper() + \
            "_HPP__\n"
        code_text += "#define __PYTHON_CONTROL_GEN_" + \
            variable_name.upper() + "_HPP__\n\n"

        code_text += f"#include \"{A_file_name}\"\n"
        code_text += f"#include \"{B_file_name}\"\n"
        code_text += f"#include \"{C_file_name}\"\n"
        code_text += f"#include \"{D_file_name}\"\n\n"
        code_text += "#include \"python_control.hpp\"\n\n"

        code_text += "namespace python_control_gen_" + variable_name + " {\n\n"

        code_text += "using namespace PythonControl;\n\n"

        code_text += f"auto A = {A_file_name_no_extension}::make();\n\n"

        code_text += f"auto B = {B_file_name_no_extension}::make();\n\n"

        code_text += f"auto C = {C_file_name_no_extension}::make();\n\n"

        code_text += f"auto D = {D_file_name_no_extension}::make();\n\n"

        code_text += f"{type_name} dt = {transfer_function.dt};\n\n"

        code_text += "using type = " + "DiscreteStateSpace_Type<" + \
            "decltype(A), decltype(B), decltype(C), decltype(D)>;\n\n"

        code_text += "inline auto make(void) -> type {\n\n"

        code_text += f"  return make_DiscreteStateSpace(A, B, C, D, dt);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace python_numpy_gen_" + variable_name + "\n\n"

        code_text += "#endif // __PYTHON_NUMPY_GEN_" + variable_name.upper() + \
            "_HPP__\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
