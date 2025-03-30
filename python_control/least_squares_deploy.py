import os
import sys
sys.path.append(os.getcwd())

import inspect
import numpy as np

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from python_control.control_deploy import ControlDeploy


class LeastSquaresDeploy:
    def __init__(self):
        pass

    @staticmethod
    def generate_LS_cpp_code(ls, file_name=None):
        deployed_file_names = []

        ControlDeploy.restrict_data_type(ls.weights.dtype.name)

        type_name = NumpyDeploy.check_dtype(ls.weights)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is ls:
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

        number_of_data = ls.number_of_data
        state_size = ls.state_size
        output_size = 1

        # %% code generation
        code_file_name = caller_file_name_without_ext + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create state-space cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += "#include \"python_numpy.hpp\"\n"
        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n\n"

        code_text += f"constexpr std::size_t LS_NUMBER_OF_DATA = {number_of_data};\n"
        code_text += f"constexpr std::size_t X_SIZE = {state_size};\n"
        code_text += f"constexpr std::size_t Y_SIZE = {output_size};\n\n"

        code_text += f"using X_Type = DenseMatrix_Type<{type_name}, "
        code_text += "LS_NUMBER_OF_DATA, X_SIZE>;\n\n"

        code_text += "using type = LeastSquares_Type<X_Type>;\n\n"

        code_text += "inline auto make(void) -> type {\n\n"

        code_text += "  return make_LeastSquares<X_Type>();\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names

    @staticmethod
    def generate_RLS_cpp_code(rls, file_name=None):
        deployed_file_names = []

        dummy_array = np.array(rls.lambda_factor)

        ControlDeploy.restrict_data_type(dummy_array.dtype.name)

        type_name = NumpyDeploy.check_dtype(dummy_array)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is rls:
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

        state_size = rls.RLS_size - 1  # remove bias term
        output_size = 1
        lambda_factor = rls.lambda_factor
        delta = rls.delta

        # %% code generation
        code_file_name = caller_file_name_without_ext + "_" + variable_name
        code_file_name_ext = code_file_name + ".hpp"

        # create state-space cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += "#include \"python_numpy.hpp\"\n"
        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n\n"

        code_text += f"constexpr std::size_t X_SIZE = {state_size};\n"
        code_text += f"constexpr std::size_t Y_SIZE = {output_size};\n\n"

        code_text += f"using X_Type = StateSpaceState_Type<{type_name}, "
        code_text += "X_SIZE>;\n\n"

        code_text += "using type = RecursiveLeastSquares_Type<X_Type>;\n\n"

        code_text += "inline auto make(void) -> type {\n\n"

        code_text += "  return make_RecursiveLeastSquares<X_Type>(\n" + \
            f"    static_cast<{type_name}>({lambda_factor}),\n" + \
            f"    static_cast<{type_name}>({delta}));\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
