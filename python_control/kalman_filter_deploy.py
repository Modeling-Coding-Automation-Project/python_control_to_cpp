import os
import sys
sys.path.append(os.getcwd())

import re
import inspect
import ast
import astor
import numpy as np
import control

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy, python_to_cpp_types
from python_control.control_deploy import ControlDeploy
from python_control.state_space_deploy import StateSpaceDeploy


class IntegerPowerReplacer(ast.NodeTransformer):
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Pow) and isinstance(node.right, ast.Constant) and isinstance(node.right.value, int) and node.right.value > 0:
            n = node.right.value
            result = node.left
            for _ in range(n - 1):
                result = ast.BinOp(left=result, op=ast.Mult(), right=node.left)
            return result
        return node

    def transform_code(self, source_code):
        tree = ast.parse(source_code)
        transformer = IntegerPowerReplacer()
        transformed_tree = transformer.visit(tree)

        transformed_code = astor.to_source(transformed_tree)

        if transformed_code.endswith("\n"):
            transformed_code = transformed_code[:-1]

        return transformed_code


class NpArrayExtractor:
    def __init__(self, code_text, Value_Type_name="float64"):
        self.code_text = code_text
        self.extract_text = ""
        self.value_type_name = Value_Type_name
        self.SparseAvailable = None

    @staticmethod
    def extract_elements(node):
        if isinstance(node, ast.List):
            return [NpArrayExtractor.extract_elements(el) for el in node.elts]
        elif isinstance(node, ast.BinOp) or isinstance(node, ast.Call) or isinstance(node, ast.Name):
            return ast.unparse(node)
        elif isinstance(node, ast.Constant):  # for Python 3.8 or later (numeric)
            return node.value
        elif isinstance(node, ast.Num):  # before Python 3.7 (numeric)
            return node.n
        elif isinstance(node, ast.UnaryOp):
            operand = NpArrayExtractor.extract_elements(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            return operand
        else:
            return node

    def extract(self):
        extract_text = ""

        matrix_content = self.code_text[self.code_text .find(
            'np.array(') + len('np.array('):-1]

        tree = ast.parse(matrix_content)

        matrix_list = NpArrayExtractor.extract_elements(tree.body[0].value)

        cols = len(matrix_list)
        rows = len(matrix_list[0]) if isinstance(
            matrix_list[0], list) else 1

        SparseAvailable = np.zeros((cols, rows), dtype=np.float64)

        for i in range(cols):
            for j in range(rows):
                if isinstance(matrix_list[i][j], (int, float)):
                    extract_text += f"result[{i}, {j}] = {matrix_list[i][j]}\n"

                    if matrix_list[i][j] != 0:
                        SparseAvailable[i, j] = True

                elif isinstance(matrix_list[i][j], str):
                    extract_text += f"result[{i}, {j}] = " + \
                        matrix_list[i][j] + "\n"

                    if matrix_list[i][j] != "0":
                        SparseAvailable[i, j] = True

        self.extract_text = extract_text
        self.SparseAvailable = SparseAvailable

    def convert_to_cpp(self):
        try:
            value_type_name = python_to_cpp_types[self.value_type_name]
        except KeyError:
            value_type_name = self.value_type_name

        pattern = r"result\[(\d+), (\d+)\] = "
        replacement = rf"result.template set<\1, \2>(static_cast<{value_type_name}>("

        convert_text = re.sub(pattern, replacement, self.extract_text)

        convert_text = convert_text.replace("\n", "));\n")

        return convert_text


class FunctionExtractor(ast.NodeVisitor):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            source_code = file.read()
        self.source_code = source_code

        self.functions = {}

    def visit_FunctionDef(self, node):
        function_name = node.name
        function_code = ast.get_source_segment(self.source_code, node)
        self.functions[function_name] = function_code
        self.generic_visit(node)

    def extract(self):
        tree = ast.parse(self.source_code)
        self.visit(tree)

        return self.functions


class InputSizeVisitor(ast.NodeVisitor):
    def __init__(self):
        self.input_size = None

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == 'INPUT_SIZE':
                if isinstance(node.value, ast.Constant):
                    self.input_size = node.value.value
                elif isinstance(node.value, ast.Num):  # For Python < 3.8
                    self.input_size = node.value.n


class FunctionToCppVisitor(ast.NodeVisitor):
    def __init__(self, Output_Type_name):
        self.cpp_code = ""
        self.Output_Type_name = Output_Type_name
        self.Value_Type_name = "double"

        self.SparseAvailable = None

    def visit_FunctionDef(self, node):
        self.cpp_code += "inline auto " + node.name + "("
        args = [arg.arg for arg in node.args.args]
        annotations = {}

        # get argument type annotations
        for arg in node.args.args:
            if arg.annotation:
                annotations[arg.arg] = ast.dump(arg.annotation)
            else:
                annotations[arg.arg] = None

        # get return type annotation
        if node.returns:
            annotations['return'] = ast.dump(node.returns)
        else:
            annotations['return'] = None

        if args[0] != "X":
            for i, arg in enumerate(args):
                Value_Type_name = self.Value_Type_name
                if annotations[arg] is not None:
                    annotation = annotations[arg]
                    Value_Type_python_name = annotation.split(
                        "attr='")[1].split("'")[0]
                    Value_Type_name = python_to_cpp_types[Value_Type_python_name]

                self.cpp_code += Value_Type_name + " " + arg
                if i == len(args) - 1:
                    break
                else:
                    self.cpp_code += ", "
        else:
            for i, arg in enumerate(args):
                type_name = annotations[arg].split(
                    "id='")[1].split("'")[0]

                if arg == "X":
                    self.cpp_code += f"{type_name} X"
                    if i == len(args) - 1:
                        break
                    else:
                        self.cpp_code += ", "
                elif arg == "U":
                    self.cpp_code += f"{type_name} U"
                    if i == len(args) - 1:
                        break
                    else:
                        self.cpp_code += ", "
                elif arg == "Parameters":
                    self.cpp_code += f"{type_name} Parameters"
                    if i == len(args) - 1:
                        break
                    else:
                        self.cpp_code += ", "

        self.cpp_code += ") -> " + self.Output_Type_name + " {\n\n"

        if node.name == "sympy_function":
            self.cpp_code += "    " + self.Output_Type_name + " result;\n\n"

        self.generic_visit(node)
        self.cpp_code += "}\n"

    def visit_Return(self, node):
        return_code = ""

        if isinstance(node.value, ast.Call):
            return_code += astor.to_source(node.value).strip()
        else:
            raise TypeError(f"Unsupported return type: {type(node.value)}")

        integer_power_replacer = IntegerPowerReplacer()
        return_code = integer_power_replacer.transform_code(return_code)

        if "np.array(" in return_code:
            np_array_extractor = NpArrayExtractor(
                return_code, self.Value_Type_name)
            np_array_extractor.extract()
            return_code = np_array_extractor.convert_to_cpp()
            self.SparseAvailable = np_array_extractor.SparseAvailable

            return_code = return_code.replace("\n", "\n    ")

            self.cpp_code += "    " + return_code + "\n"
            self.cpp_code += "    return result;\n"
        else:
            self.cpp_code += "    return " + return_code + ";\n"

    def visit_Assign(self, node):
        assign_code = ""
        targets = [astor.to_source(t).strip() for t in node.targets]
        value = astor.to_source(node.value).strip()
        assign_code += "    " + self.Value_Type_name + " " + \
            ", ".join(targets) + " = " + value + ";\n"
        assign_code += "\n"

        assign_code = assign_code.replace("[", ".template get<")
        assign_code = assign_code.replace("]", ">()")

        self.cpp_code += assign_code

    def convert(self, python_code):
        tree = ast.parse(python_code)
        self.visit(tree)

        return self.cpp_code


class KalmanFilterDeploy:
    def __init__(self):
        pass

    @staticmethod
    def create_sympy_code(sym_object):
        value_example = np.array([[1.0]])
        value_type = str(value_example.dtype.name)

        code_text = ""
        code_text += "def sympy_function("

        arguments_text = ""
        arguments_text_out = ""

        sym_symbols = sym_object.free_symbols
        for i, symbol in enumerate(sym_symbols):
            arguments_text += f"{symbol}"
            arguments_text_out += f"{symbol}"
            if i == len(sym_symbols) - 1:
                arguments_text += f": np.{value_type}"
                break
            else:
                arguments_text += f": np.{value_type}" + ", "
                arguments_text_out += ", "

        code_text += arguments_text + ")"

        code_text += f" -> Tuple[{sym_object.shape[0]}, {sym_object.shape[1]}]:\n\n"

        calculation_code = f"{sym_object.tolist()}"

        code_text += f"    return np.array({calculation_code})\n\n\n"

        return code_text, arguments_text_out

    @staticmethod
    def create_interface_code(sym_object, arguments_text, X, U=None):
        sym_symbols = sym_object.free_symbols

        code_text = ""
        code_text += "def function(X: X_Type"

        if U is not None:
            code_text += ", U: U_Type, Parameters: Parameter_Type = None)"
        else:
            code_text += ", Parameters: Parameter_Type = None)"

        code_text += f" -> Tuple[{sym_object.shape[0]}, {sym_object.shape[1]}]:\n\n"

        for i in range(X.shape[0]):
            if X[i] in sym_symbols:
                code_text += f"    {X[i]} = X[{i}, 0]\n"
                sym_symbols.remove(X[i])

        if U is not None:
            for i in range(U.shape[0]):
                if U[i] in sym_symbols:
                    code_text += f"    {U[i]} = U[{i}, 0]\n"
                    sym_symbols.remove(U[i])

        code_text += "\n"

        for symbol in sym_symbols:
            code_text += f"    {symbol} = "
            code_text += f"Parameters.{symbol}\n"

        code_text += "\n"

        code_text += "    return sympy_function("
        code_text += arguments_text
        code_text += ")\n"

        return code_text

    @staticmethod
    def write_function_code_from_sympy(sym_object, sym_object_name, X, U=None):
        header_code = ""
        header_code += "import numpy as np\n"
        header_code += "from math import *\n"
        header_code += "from typing import Tuple\n\n\n"

        header_code += "class X_Type:\n    pass\n\n\n"

        header_code += "class U_Type:\n    pass\n\n\n"

        header_code += "class Parameter_Type:\n    pass\n\n\n"

        header_code += "STATE_SIZE = " + str(X.shape[0]) + "\n"
        if U is not None:
            header_code += "INPUT_SIZE = " + str(U.shape[0])
        else:
            header_code += "INPUT_SIZE = 0"
        header_code += "\n\n\n"

        sympy_function_code, arguments_text = KalmanFilterDeploy.create_sympy_code(
            sym_object)

        interface_code = KalmanFilterDeploy.create_interface_code(
            sym_object, arguments_text, X, U)

        total_code = header_code + sympy_function_code + interface_code

        ControlDeploy.write_to_file(
            total_code, f"{sym_object_name}.py")

    @staticmethod
    def write_state_function_code_from_sympy(sym_object, X, U=None):
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        sym_object_name = None
        for name, value in caller_locals.items():
            if value is sym_object:
                sym_object_name = name
                break

        KalmanFilterDeploy.write_function_code_from_sympy(
            sym_object, sym_object_name, X, U)

    @staticmethod
    def write_measurement_function_code_from_sympy(sym_object, X):
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        sym_object_name = None
        for name, value in caller_locals.items():
            if value is sym_object:
                sym_object_name = name
                break

        KalmanFilterDeploy.write_function_code_from_sympy(
            sym_object, sym_object_name, X, U=None)

    @staticmethod
    def get_input_size_from_function_code(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            file_content = file.read()

        tree = ast.parse(file_content)
        visitor = InputSizeVisitor()
        visitor.visit(tree)

        return visitor.input_size

    @staticmethod
    def find_file(filename, search_path):
        for root, _, files in os.walk(search_path):
            if filename in files:
                return os.path.join(root, filename)
        return None

    @staticmethod
    def generate_parameter_cpp_code(parameter_object, value_type_name):
        try:
            value_type_name = python_to_cpp_types[value_type_name]
        except KeyError:
            pass

        code_text = ""
        code_text += "class Parameter {\n"
        code_text += "public:\n"

        elements = parameter_object.__dict__
        for key, value in elements.items():
            code_text += f"  {value_type_name} {key} = static_cast<{value_type_name}>({value});\n"

        code_text += "};\n\n"

        code_text += "using Parameter_Type = Parameter;\n"

        return code_text

    @staticmethod
    def generate_LKF_cpp_code(lkf, file_name=None):
        deployed_file_names = []

        ControlDeploy.restrict_data_type(lkf.A.dtype.name)

        type_name = NumpyDeploy.check_dtype(lkf.A)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is lkf:
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

        # create state space from A, B, C matrices
        lkf_D = np.zeros((lkf.C.shape[0], lkf.B.shape[1]))
        dt = 1.0  # dt is not used in LinearKalmanFilter calculation.

        exec(f"{variable_name}_ss = control.ss(lkf.A, lkf.B, lkf.C, lkf_D, dt)")
        ss_file_names = eval(
            f"StateSpaceDeploy.generate_state_space_cpp_code({variable_name}_ss, caller_file_name_without_ext)")

        deployed_file_names.append(ss_file_names)
        ss_file_name = ss_file_names[-1]

        ss_file_name_no_extension = ss_file_name.split(".")[0]

        # create cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{ss_file_name}\"\n\n"
        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n\n"

        code_text += f"auto lkf_state_space = {ss_file_name_no_extension}::make();\n\n"

        code_text += "constexpr std::size_t STATE_SIZE = decltype(lkf_state_space.A)::COLS;\n"
        code_text += "constexpr std::size_t INPUT_SIZE = decltype(lkf_state_space.B)::ROWS;\n"
        code_text += "constexpr std::size_t OUTPUT_SIZE = decltype(lkf_state_space.C)::COLS;\n\n"

        code_text += "auto Q = make_DiagMatrix<STATE_SIZE>(\n"
        for i in range(lkf.Q.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(lkf.Q[i, i]) + ")"
            if i == lkf.Q.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += ");\n\n"

        code_text += "auto R = make_DiagMatrix<OUTPUT_SIZE>(\n"
        for i in range(lkf.R.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(lkf.R[i, i]) + ")"
            if i == lkf.R.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += ");\n\n"

        code_text += "using type = LinearKalmanFilter_Type<" + \
            "decltype(lkf_state_space), decltype(Q), decltype(R)>;\n\n"

        code_text += "auto make() -> type {\n\n"

        code_text += "    return make_LinearKalmanFilter(lkf_state_space, Q, R);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names

    @staticmethod
    def generate_EKF_cpp_code(ekf, file_name=None):
        deployed_file_names = []

        ControlDeploy.restrict_data_type(ekf.A.dtype.name)

        type_name = NumpyDeploy.check_dtype(ekf.A)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is ekf:
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

        # generate parameter class code
        parameter_code = KalmanFilterDeploy.generate_parameter_cpp_code(
            ekf.Parameters, type_name)

        # create state and measurement functions
        fxu_name = ekf.state_function.__module__
        state_function_code, state_function_U_size, _ = \
            KalmanFilterDeploy.create_state_and_measurement_function_code(ekf, fxu_name,
                                                                          "X_Type")

        fxu_jacobian_name = ekf.state_function_jacobian.__module__
        state_function_jacobian_code, _, A_SparseAvailable_list = \
            KalmanFilterDeploy.create_state_and_measurement_function_code(ekf, fxu_jacobian_name,
                                                                          "A_Type")
        ekf_A = A_SparseAvailable_list[0]

        hx_name = ekf.measurement_function.__module__
        measurement_function_code, _, _ = \
            KalmanFilterDeploy.create_state_and_measurement_function_code(ekf, hx_name,
                                                                          "Y_Type")

        hx_jacobian_name = ekf.measurement_function_jacobian.__module__
        measurement_function_jacobian_code, _, C_SparseAvailable_list = \
            KalmanFilterDeploy.create_state_and_measurement_function_code(ekf, hx_jacobian_name,
                                                                          "C_Type")
        ekf_C = C_SparseAvailable_list[0]

        # create A, C matrices
        exec(f"{variable_name}_A = ekf_A")
        A_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_A, caller_file_name_without_ext)")
        exec(f"{variable_name}_C = ekf_C")
        C_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code({variable_name}_C, caller_file_name_without_ext)")

        deployed_file_names.append(A_file_name)
        deployed_file_names.append(C_file_name)

        A_file_name_no_extension = A_file_name.split(".")[0]
        C_file_name_no_extension = C_file_name.split(".")[0]

        # create cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{A_file_name}\"\n"
        code_text += f"#include \"{C_file_name}\"\n\n"

        code_text += "#include \"python_math.hpp\"\n"
        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n\n"

        code_text += f"using A_Type = {A_file_name_no_extension}::type;\n"
        code_text += f"auto A = {A_file_name_no_extension}::make();\n\n"

        code_text += f"using C_Type = {C_file_name_no_extension}::type;\n"
        code_text += f"auto C = {C_file_name_no_extension}::make();\n\n"

        code_text += "constexpr std::size_t STATE_SIZE = A_Type::COLS;\n"
        code_text += f"constexpr std::size_t INPUT_SIZE = {state_function_U_size};\n"
        code_text += "constexpr std::size_t OUTPUT_SIZE = C_Type::COLS;\n\n"

        code_text += "using X_Type = StateSpaceStateType<double, STATE_SIZE>;\n"
        code_text += "using U_Type = StateSpaceInputType<double, INPUT_SIZE>;\n"
        code_text += "using Y_Type = StateSpaceOutputType<double, OUTPUT_SIZE>;\n\n"

        code_text += parameter_code
        code_text += "\n"

        code_text += "namespace state_function {\n\n"

        code_text += "using namespace PythonMath;\n\n"

        for code in state_function_code:
            code_text += code
            code_text += "\n"

        code_text += "} // namespace state_function\n\n"

        code_text += "namespace state_function_jacobian {\n\n"

        code_text += "using namespace PythonMath;\n\n"

        for code in state_function_jacobian_code:
            code_text += code
            code_text += "\n"

        code_text += "} // namespace state_function_jacobian\n\n"

        code_text += "namespace measurement_function {\n\n"

        code_text += "using namespace PythonMath;\n\n"

        for code in measurement_function_code:
            code_text += code
            code_text += "\n"

        code_text += "} // namespace measurement_function\n\n"

        code_text += "namespace measurement_function_jacobian {\n\n"

        code_text += "using namespace PythonMath;\n\n"

        for code in measurement_function_jacobian_code:
            code_text += code
            code_text += "\n"

        code_text += "} // namespace measurement_function_jacobian\n\n"

        code_text += "auto Q = make_DiagMatrix<STATE_SIZE>(\n"
        for i in range(ekf.Q.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(ekf.Q[i, i]) + ")"
            if i == ekf.Q.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += ");\n\n"

        code_text += "auto R = make_DiagMatrix<OUTPUT_SIZE>(\n"
        for i in range(ekf.R.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(ekf.R[i, i]) + ")"
            if i == ekf.R.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += ");\n\n"

        code_text += "using type = ExtendedKalmanFilter_Type<" + \
            "A_Type, C_Type, U_Type, decltype(Q), decltype(R), Parameter_Type>;\n\n"

        code_text += "auto make() -> type {\n\n"

        code_text += "    Parameter_Type parameters;\n\n"

        code_text += "    StateFunction_Object<X_Type, U_Type, Parameter_Type> state_function_object =\n" + \
            "        state_function::function;\n\n"

        code_text += "    StateFunctionJacobian_Object<A_Type, X_Type, U_Type, Parameter_Type> state_function_jacobian_object =\n" + \
            "        state_function_jacobian::function;\n\n"

        code_text += "    MeasurementFunction_Object<Y_Type, X_Type, Parameter_Type> measurement_function_object =\n" + \
            "        measurement_function::function;\n\n"

        code_text += "    MeasurementFunctionJacobian_Object<C_Type, X_Type, Parameter_Type> measurement_function_jacobian_object =\n" + \
            "        measurement_function_jacobian::function;\n\n"

        code_text += "    return ExtendedKalmanFilter_Type<\n" + \
            "        A_Type, C_Type, U_Type, decltype(Q), decltype(R), Parameter_Type>(\n" + \
            "        Q, R, state_function_object, state_function_jacobian_object,\n" + \
            "        measurement_function_object, measurement_function_jacobian_object,\n" + \
            "        parameters);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names

    @staticmethod
    def create_state_and_measurement_function_code(ekf, function_name, return_type):
        fxu_file_path = KalmanFilterDeploy.find_file(
            f"{function_name}.py", os.getcwd())
        state_function_U_size = KalmanFilterDeploy.get_input_size_from_function_code(
            fxu_file_path)

        extractor = FunctionExtractor(fxu_file_path)
        functions = extractor.extract()
        state_function_code = []
        SparseAvailable_list = []

        for name, code in functions.items():
            converter = FunctionToCppVisitor(return_type)

            state_function_code.append(converter.convert(code))
            SparseAvailable_list.append(converter.SparseAvailable)

        SparseAvailable_list = [
            x for x in SparseAvailable_list if x is not None]

        return state_function_code, state_function_U_size, SparseAvailable_list
