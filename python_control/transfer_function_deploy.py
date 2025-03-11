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
        den_factors = transfer_function.den[0][0]
        num_factors = transfer_function.num[0][0]

        ControlDeploy.restrict_data_type(
            den_factors.dtype.name)

        type_name = NumpyDeploy.check_dtype(den_factors)

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

        # create state-space cpp code
        code_text = ""

        file_header_macro_name = "__PYTHON_CONTROL_GEN_" + variable_name.upper() + \
            "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = "namespace python_control_gen_" + variable_name

        code_text += namespace_name + " {\n\n"

        code_text += "using namespace PythonControl;\n\n"

        code_text += f"auto numerator = make_TransferFunctionNumerator<{num_factors.shape[0]}>(\n"

        for i in range(num_factors.shape[0]):
            code_text += f"  static_cast<{type_name}>({num_factors[i]})"
            if i < num_factors.shape[0] - 1:
                code_text += ",\n"
            else:
                code_text += "\n);\n\n"

        code_text += f"auto denominator = make_TransferFunctionDenominator<{den_factors.shape[0]}>(\n"

        for i in range(den_factors.shape[0]):
            code_text += f"  static_cast<{type_name}>({den_factors[i]})"
            if i < den_factors.shape[0] - 1:
                code_text += ",\n"
            else:
                code_text += "\n);\n\n"

        code_text += f"{type_name} dt = static_cast<{type_name}>({transfer_function.dt});\n\n"

        code_text += "using type = " + "DiscreteTransferFunction<" + \
            "decltype(numerator), decltype(denominator)>;\n\n"

        code_text += "inline auto make(void) -> type {\n\n"

        code_text += f"  return make_DiscreteTransferFunction(numerator, denominator, dt);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
