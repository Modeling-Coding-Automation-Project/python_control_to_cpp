import os
import sys
sys.path.append(os.getcwd())

import inspect

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from python_control.control_deploy import ControlDeploy


class StateSpaceDeploy:
    def __init__(self):
        pass

    @staticmethod
    def generate_state_space_cpp_code(state_space, file_name=None, delay_step=0):
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

        code_text += f"constexpr std::size_t DELAY_STEP = {delay_step};\n\n"

        code_text += f"auto A = {A_file_name_no_extension}::make();\n\n"

        code_text += f"auto B = {B_file_name_no_extension}::make();\n\n"

        code_text += f"auto C = {C_file_name_no_extension}::make();\n\n"

        code_text += f"auto D = {D_file_name_no_extension}::make();\n\n"

        code_text += f"{type_name} dt = static_cast<{type_name}>({state_space.dt});\n\n"

        code_text += "using type = DiscreteStateSpace_Type<\n" + \
            "    decltype(A), decltype(B), decltype(C), decltype(D), DELAY_STEP>;\n\n"

        code_text += "inline auto make(void) -> type {\n\n"

        code_text += f"  return make_DiscreteStateSpace<DELAY_STEP>(\n" + \
            "    A, B, C, D, dt);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
