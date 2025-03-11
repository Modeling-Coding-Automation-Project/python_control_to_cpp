import os
import sys
sys.path.append(os.getcwd())

import copy
import numpy as np
import control

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from python_control.control_deploy import ControlDeploy
from python_control.state_space_deploy import StateSpaceDeploy


class LQR_Deploy:
    def __init__(self):
        pass

    @staticmethod
    def generate_LQR_cpp_code(Ac, Bc, Q, R):
        deployed_file_names = []

        ControlDeploy.restrict_data_type(Ac.dtype.name)

        type_name = NumpyDeploy.check_dtype(Ac)

        variable_name = "LQR"

        code_file_name = "python_control_gen_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create Ac, Bc matrices
        exec(f"{variable_name}_Ac = copy.deepcopy(Ac)")
        Ac_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Ac)")
        exec(f"{variable_name}_Bc = copy.deepcopy(Bc)")
        Bc_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_Bc)")

        deployed_file_names.append(Ac_file_name)
        deployed_file_names.append(Bc_file_name)

        Ac_file_name_no_extension = Ac_file_name.split(".")[0]
        Bc_file_name_no_extension = Bc_file_name.split(".")[0]

        # create state-space cpp code
        code_text = ""

        file_header_macro_name = "__PYTHON_CONTROL_GEN_" + variable_name.upper() + \
            "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{Ac_file_name}\"\n"
        code_text += f"#include \"{Bc_file_name}\"\n\n"
        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = "python_control_gen_" + variable_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n\n"

        code_text += f"auto Ac = {Ac_file_name_no_extension}::make();\n\n"

        code_text += f"auto Bc = {Bc_file_name_no_extension}::make();\n\n"

        code_text += "constexpr std::size_t STATE_SIZE = decltype(Ac)::COLS;\n"
        code_text += "constexpr std::size_t INPUT_SIZE = decltype(Bc)::ROWS;\n\n"

        code_text += "auto Q = make_DiagMatrix<STATE_SIZE>(\n"
        for i in range(Q.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(Q[i, i]) + ")"
            if i == Q.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += ");\n\n"

        code_text += "auto R = make_DiagMatrix<INPUT_SIZE>(\n"
        for i in range(R.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(R[i, i]) + ")"
            if i == R.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ", \n"
        code_text += ");\n\n"

        code_text += "using type = LQR_Type<" + \
            "decltype(Ac), decltype(Bc), decltype(Q), decltype(R)>;\n\n"

        code_text += "auto make() -> type {\n\n"

        code_text += "    return make_LQR(Ac, Bc, Q, R);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
