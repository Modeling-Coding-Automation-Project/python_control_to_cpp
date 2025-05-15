import os
import sys
sys.path.append(os.getcwd())

import inspect
import control

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy
from python_control.control_deploy import ControlDeploy


class TransferFunctionDeploy(ControlDeploy):
    def __init__(self):
        super().__init__()

    @staticmethod
    def generate_transfer_function_cpp_code(
            transfer_function: control.TransferFunction, file_name=None, number_of_delay=0):
        deployed_file_names = []
        den_factors = transfer_function.den[0][0]
        num_factors = transfer_function.num[0][0]

        ControlDeploy.restrict_data_type(
            den_factors.dtype.name)

        type_name = NumpyDeploy.check_dtype(den_factors)

        # %% inspect arguments
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

        # create state-space cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonControl;\n\n"

        code_text += f"constexpr std::size_t NUMERATOR_SIZE = {num_factors.shape[0]};\n"
        code_text += f"constexpr std::size_t DENOMINATOR_SIZE = {den_factors.shape[0]};\n"
        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {number_of_delay};\n\n"

        code_text += f"using Numerator_Type = TransferFunctionNumerator_Type<{type_name}, NUMERATOR_SIZE>;\n\n"

        code_text += f"using Denominator_Type = TransferFunctionDenominator_Type<{type_name}, DENOMINATOR_SIZE>;\n\n"

        code_text += "using type = " + "DiscreteTransferFunction<\n" + \
            "    Numerator_Type, Denominator_Type, NUMBER_OF_DELAY>;\n\n"

        code_text += "inline auto make(void) -> type {\n\n"

        code_text += "  auto numerator = make_TransferFunctionNumerator<NUMERATOR_SIZE>(\n"

        for i in range(num_factors.shape[0]):
            code_text += f"    static_cast<{type_name}>({num_factors[i]})"
            if i < num_factors.shape[0] - 1:
                code_text += ",\n"
            else:
                code_text += "\n  );\n\n"

        code_text += "  auto denominator = make_TransferFunctionDenominator<DENOMINATOR_SIZE>(\n"

        for i in range(den_factors.shape[0]):
            code_text += f"    static_cast<{type_name}>({den_factors[i]})"
            if i < den_factors.shape[0] - 1:
                code_text += ",\n"
            else:
                code_text += "\n  );\n\n"

        code_text += f"  {type_name} dt = static_cast<{type_name}>({transfer_function.dt});\n\n"

        code_text += f"  return make_DiscreteTransferFunction<NUMBER_OF_DELAY>(\n" + \
            "    numerator, denominator, dt);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
