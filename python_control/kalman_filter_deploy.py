"""
This module provides the `KalmanFilterDeploy` class,
which contains a comprehensive set of static methods for automating the
deployment and code generation of Kalman filter variants
(Linear, Extended, and Unscented Kalman Filters) from Python to C++ code.
The class is designed to facilitate the conversion of symbolic and numerical
representations of system models and filter parameters into deployable C++ header files,
supporting rapid prototyping and integration of control and estimation algorithms.
"""
import os
import sys
sys.path.append(os.getcwd())

import re
import inspect
import ast
import astor
import numpy as np
import sympy as sp
import control
from dataclasses import dataclass, fields, is_dataclass

from external_libraries.MCAP_python_control.python_control.kalman_filter import LinearKalmanFilter
from external_libraries.MCAP_python_control.python_control.kalman_filter import ExtendedKalmanFilter
from external_libraries.MCAP_python_control.python_control.kalman_filter import UnscentedKalmanFilter

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy, python_to_cpp_types
from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import FunctionExtractor
from external_libraries.MCAP_python_control.python_control.control_deploy import IntegerPowerReplacer
from external_libraries.MCAP_python_control.python_control.control_deploy import InputSizeVisitor
from python_control.state_space_deploy import StateSpaceDeploy


class NpArrayExtractor:
    """
    A class to extract elements from a NumPy array defined in Python code and convert them into C++ code.
    This class parses the provided Python code to find NumPy array definitions, extracts their elements,
    and generates C++ code that initializes a result array with the extracted values.
    Attributes:
        code_text (str): The Python code containing the NumPy array definition.
        extract_text (str): The extracted C++ code for initializing the result array.
        value_type_name (str): The C++ type name for the values in the NumPy array.
        SparseAvailable (np.ndarray): A NumPy array indicating the sparsity of the extracted values.
    """

    def __init__(self, code_text, Value_Type_name="float64"):
        self.code_text = code_text
        self.extract_text = ""
        self.value_type_name = Value_Type_name
        self.SparseAvailable = None

    @staticmethod
    def extract_elements(node):
        """
        Recursively extracts elements from an AST node representing a NumPy array or similar structure.
        Args:
            node (ast.AST): The AST node to extract elements from.
        Returns:
            list or str: A list of extracted elements if the node is a list, or a string representation of the node.
        If the node is a constant numeric value, it returns the value directly.
        If the node is a unary operation, it returns the negated value of the operand.
        Raises:
            TypeError: If the node type is not supported for extraction.
        """

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
                if isinstance(operand, (int, float)):
                    return -operand
                elif isinstance(operand, str):
                    return f"-{operand}"
            return operand
        else:
            return node

    def extract(self):
        """
        Extracts elements from the NumPy array defined in the provided Python code and generates C++ code
        to initialize a result array with the extracted values.
        This method parses the Python code to find the NumPy array definition, extracts its elements,
        and constructs C++ code that initializes a result array with the extracted values.
        It also creates a NumPy array indicating the sparsity of the extracted values.
        The generated C++ code is stored in the `extract_text` attribute, and the sparsity information
        is stored in the `SparseAvailable` attribute.
        """
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
        """
        Converts the extracted Python code into C++ code that initializes a result array with the extracted values.
        This method replaces the Python-specific syntax for array initialization with C++ syntax,
        ensuring that the resulting C++ code is syntactically correct and compatible with C++ standards.
        It also handles the conversion of Python data types to their C++ equivalents based on the specified
        value type name.
        Returns:
            str: The converted C++ code that initializes the result array with the extracted values.
        """
        try:
            value_type_name = python_to_cpp_types[self.value_type_name]
        except KeyError:
            value_type_name = self.value_type_name

        pattern = r"result\[(\d+), (\d+)\] = "
        replacement = rf"result.template set<\1, \2>(static_cast<{value_type_name}>("

        convert_text = re.sub(pattern, replacement, self.extract_text)

        convert_text = convert_text.replace("\n", "));\n")

        return convert_text


class FunctionToCppVisitor(ast.NodeVisitor):
    """
    A class to convert Python function definitions into C++ function definitions.
    This class traverses the abstract syntax tree (AST) of Python code, specifically looking for function definitions,
    and generates C++ code for those functions. It handles type annotations, return types, and the conversion of
    NumPy array operations into C++ syntax.
    Attributes:
        cpp_code (str): The generated C++ code for the function definitions.
        Output_Type_name (str): The name of the C++ type for the function's output.
        Value_Type_name (str): The name of the C++ type for the function's input values.
        SparseAvailable (bool or None): Indicates whether sparse matrix operations are available in the function.
    """

    def __init__(self, Output_Type_name):
        self.cpp_code = ""
        self.Output_Type_name = Output_Type_name
        self.Value_Type_name = "double"

        self.SparseAvailable = None

    def visit_FunctionDef(self, node):
        """
        Visits function definition nodes in the AST and generates C++ code for them.
        Args:
            node (ast.FunctionDef): The function definition node to visit.
        This method constructs the C++ function signature, including the function name, argument types,
        and return type. It also handles type annotations for each argument and the return type.
        It generates the C++ code for the function body by visiting the function's body nodes.
        The generated C++ code is stored in the `cpp_code` attribute.
        If the function name is "sympy_function", it initializes a result variable of type `Output_Type_name`.
        If the first argument is "X", it handles special cases for the arguments "X", "U", and "Parameters".
        """
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

        if args != []:
            if args[0] != "X":
                for i, arg in enumerate(args):
                    Value_Type_name = self.Value_Type_name
                    if annotations[arg] is not None:
                        annotation = annotations[arg]
                        Value_Type_python_name = annotation.split(
                            "attr='")[1].split("'")[0]
                        Value_Type_name = python_to_cpp_types[Value_Type_python_name]

                    self.cpp_code += "const " + Value_Type_name + " " + arg
                    if i == len(args) - 1:
                        break
                    else:
                        self.cpp_code += ", "
            else:
                for i, arg in enumerate(args):
                    type_name = annotations[arg].split(
                        "id='")[1].split("'")[0]

                    if arg == "X":
                        self.cpp_code += f"const {type_name} X"
                        if i == len(args) - 1:
                            break
                        else:
                            self.cpp_code += ", "
                    elif arg == "U":
                        self.cpp_code += f"const {type_name} U"
                        if i == len(args) - 1:
                            break
                        else:
                            self.cpp_code += ", "
                    elif arg == "Parameters":
                        self.cpp_code += f"const {type_name} Parameters"
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
        """
        Visits return nodes in the AST and generates C++ code for the return statement.
        Args:
            node (ast.Return): The return node to visit.
        This method constructs the C++ code for the return statement based on the value being returned.
        If the return value is a function call, it converts it to C++ syntax using `astor.to_source`.
        If the return value is a NumPy array, it extracts the elements using `NpArrayExtractor` and converts them to C++ code.
        It also handles integer power operations by replacing them with repeated multiplications using `IntegerPowerReplacer`.
        The generated C++ code is appended to the `cpp_code` attribute.
        If the return value is not supported, it raises a `TypeError`.
        """
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
        """
        Visits assignment nodes in the AST and generates C++ code for the assignment.
        Args:
            node (ast.Assign): The assignment node to visit.
        This method constructs the C++ code for the assignment statement by extracting the target variables
        and the value being assigned. It converts the target variables to C++ syntax using `astor.to_source`
        and the value to C++ syntax as well. The generated C++ code is appended to the `cpp_code` attribute.
        The assignment code is formatted to match C++ syntax, including the use of `template get<>` for array indexing.
        The generated code is indented and formatted to ensure proper C++ syntax.
        The assignment code is constructed by iterating over the targets and joining them with commas.
        The value is also converted to C++ syntax using `astor.to_source`.
        The generated C++ code is appended to the `cpp_code` attribute, and it replaces Python list indexing
        with C++ template syntax for accessing elements.
        """
        integer_power_replacer = IntegerPowerReplacer()
        assign_code = ""

        targets = [astor.to_source(t).strip() for t in node.targets]
        value = astor.to_source(node.value).strip()
        value = integer_power_replacer.transform_code(value)

        assign_code += "    " + self.Value_Type_name + " " + \
            ", ".join(targets) + " = " + value + ";\n"
        assign_code += "\n"

        assign_code = assign_code.replace("[", ".template get<")
        assign_code = assign_code.replace("]", ">()")

        self.cpp_code += assign_code

    def convert(self, python_code):
        """
        Converts the provided Python code into C++ code by parsing it into an AST and visiting each node.
        Args:
            python_code (str): The Python code to convert to C++.
        Returns:
            str: The generated C++ code.
        This method initializes the AST parser with the provided Python code, creates an instance of the
        `FunctionToCppVisitor`, and visits the AST nodes to generate the C++ code.
        It uses the `ast.parse` function to parse the Python code into an AST, and then calls the `visit` method
        of the `FunctionToCppVisitor` instance to traverse the AST and generate the C++ code.
        The generated C++ code is stored in the `cpp_code` attribute of the `FunctionToCppVisitor` instance.
        Returns:
            str: The generated C++ code.
        """
        tree = ast.parse(python_code)
        self.visit(tree)

        return self.cpp_code


class KalmanFilterDeploy:
    """
    A class for deploying Kalman filter algorithms by generating C++ code from symbolic representations.
    This class provides methods to create C++ code for Kalman filter state and measurement functions,
    as well as to generate C++ code for Linear, Extended, and Unscented Kalman Filters.
    It includes functionality to convert symbolic expressions into C++ functions, handle input and output types,
    and write the generated code to files. The class also supports the generation of parameter classes for
    Kalman filters, allowing for easy integration of filter parameters into the generated C++ code.
    """

    def __init__(self):
        pass

    @staticmethod
    def generate_parameter_cpp_code(parameter_object, value_type_name: str):
        """
        Generates C++ code for a parameter class based on a Python object.
        Args:
            parameter_object (object): The Python object representing the parameters.
            value_type_name (str): The C++ type name for the parameter values.
        Returns:
            str: The generated C++ code for the parameter class.
        This method inspects the attributes of the provided parameter object and generates a C++ class definition
        that contains member variables corresponding to the attributes of the parameter object.
        It converts the Python data types to their C++ equivalents based on the provided value type name.
        If the value type name is not found in the predefined mapping, it leaves the type name unchanged.
        The generated C++ code includes a class definition with public member variables initialized to the values
        from the parameter object. It also defines a type alias for the parameter class.
        """
        if not is_dataclass(parameter_object):
            raise TypeError("parameter_object must be a dataclass instance")

        try:
            value_type_name = python_to_cpp_types[value_type_name]
        except KeyError:
            pass

        code_text = ""
        code_text += "class Parameter {\n"
        code_text += "public:\n"

        name_value_pairs = [(f.name, getattr(parameter_object, f.name))
                            for f in fields(parameter_object)]
        for i, name_value in enumerate(name_value_pairs):
            code_text += f"  {value_type_name} {name_value[0]} = static_cast<{value_type_name}>({name_value[1]});\n"

        code_text += "};\n\n"

        code_text += "using Parameter_Type = Parameter;\n"

        return code_text

    @staticmethod
    def create_cpp_code_file(file_name_without_ext: str,
                             code_suffix_text: str,
                             code_body_text: str):
        code_text = ""

        code_text += "#ifndef __" + file_name_without_ext.upper() + "_HPP__\n"
        code_text += "#define __" + file_name_without_ext.upper() + "_HPP__\n\n"

        code_text += code_suffix_text

        code_text += "namespace " + file_name_without_ext + " {\n\n"

        code_text += code_body_text + "\n"

        code_text += "} // namespace " + file_name_without_ext + "\n\n"

        code_text += "#endif // __" + file_name_without_ext.upper() + "_HPP__\n"

        ControlDeploy.write_to_file(code_text, file_name_without_ext + ".hpp")

    @staticmethod
    def generate_LKF_cpp_code(lkf, file_name=None, number_of_delay=0):
        """
        Generates C++ code for a Linear Kalman Filter (LKF) based on the provided LKF object.
        Args:
            lkf (LinearKalmanFilter): An instance of the LinearKalmanFilter class containing the filter parameters.
            file_name (str, optional): The name of the file to which the generated code will be written. Defaults to None.
            number_of_delay (int, optional): The number of delays to consider in the filter. Defaults to 0.
        Returns:
            list: A list of file names where the generated C++ code is written.
        This method generates C++ code for a Linear Kalman Filter by inspecting the provided LKF object.
        It restricts the data type of the filter, checks the data type of the A matrix, and generates C++ code
        for the state space representation of the filter. It also creates a C++ header file containing the
        necessary includes, namespace definitions, and type definitions for the filter.
        The generated code includes the initialization of the filter's state space, covariance matrices, and
        the Kalman gain. It also handles the initialization of the filter's parameters if they are provided.
        The generated C++ code is written to a file named `<caller_file_name_without_ext>_<variable_name>.hpp`,
        where `<caller_file_name_without_ext>` is the name of the file from which the function is called,
        and `<variable_name>` is the name of the LKF variable in the caller's local scope.
        Raises:
            ValueError: If the data type of the A matrix is not supported.
        """

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
            f"StateSpaceDeploy.generate_state_space_cpp_code({variable_name}_ss, caller_file_name_without_ext, number_of_delay={number_of_delay})")

        deployed_file_names.append(ss_file_names)
        ss_file_name = ss_file_names[-1]

        ss_file_name_no_extension = ss_file_name.split(".")[0]

        # generate P, G initialization code
        P_G_initialization_flag = False
        if lkf.G is not None:
            P_G_initialization_flag = True

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

        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {number_of_delay};\n\n"

        code_text += f"using LkfStateSpace_Type = {ss_file_name_no_extension}::type;\n\n"

        code_text += "constexpr std::size_t STATE_SIZE = LkfStateSpace_Type::A_Type::COLS;\n"
        code_text += "constexpr std::size_t INPUT_SIZE = LkfStateSpace_Type::B_Type::ROWS;\n"
        code_text += "constexpr std::size_t OUTPUT_SIZE = LkfStateSpace_Type::C_Type::COLS;\n\n"

        code_text += f"using Q_Type = KalmanFilter_Q_Type<{type_name}, STATE_SIZE>;\n\n"

        code_text += f"using R_Type = KalmanFilter_R_Type<{type_name}, OUTPUT_SIZE>;\n\n"

        code_text += "using type = LinearKalmanFilter_Type<\n" + \
            "    LkfStateSpace_Type, Q_Type, R_Type>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += f"  auto lkf_state_space = {ss_file_name_no_extension}::make();\n\n"

        code_text += "  auto Q = make_KalmanFilter_Q<STATE_SIZE>(\n"
        for i in range(lkf.Q.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(lkf.Q[i, i]) + ")"
            if i == lkf.Q.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += "  );\n\n"

        code_text += "  auto R = make_KalmanFilter_R<OUTPUT_SIZE>(\n"
        for i in range(lkf.R.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(lkf.R[i, i]) + ")"
            if i == lkf.R.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += "  );\n\n"

        code_text += "  auto lkf = make_LinearKalmanFilter(\n" + \
            "    lkf_state_space, Q, R);\n\n"

        if P_G_initialization_flag:
            code_text += "  lkf.P = make_DenseMatrix<STATE_SIZE, STATE_SIZE>(\n"
            for i in range(lkf.P.shape[0]):
                for j in range(lkf.P.shape[1]):
                    code_text += "    static_cast<" + \
                        type_name + ">(" + str(lkf.P[i, j]) + ")"
                    if i == lkf.P.shape[0] - 1 and j == lkf.P.shape[1] - 1:
                        code_text += "\n"
                        break
                    else:
                        code_text += ",\n"
            code_text += "  );\n\n"

            code_text += "  lkf.G = make_DenseMatrix<STATE_SIZE, OUTPUT_SIZE>(\n"
            for i in range(lkf.G.shape[0]):
                for j in range(lkf.G.shape[1]):
                    code_text += "    static_cast<" + \
                        type_name + ">(" + str(lkf.G[i, j]) + ")"
                    if i == lkf.G.shape[0] - 1 and j == lkf.G.shape[1] - 1:
                        code_text += "\n"
                        break
                    else:
                        code_text += ",\n"
            code_text += "  );\n\n"

        code_text += "  return lkf;\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names

    @staticmethod
    def create_state_and_measurement_function_code(function_name, return_type):
        """
        Creates the state and measurement function code from a Python function file.
        Args:
            function_name (str): The name of the function to extract code from.
            return_type (str): The return type of the function, used for type conversion.
        Returns:
            tuple: A tuple containing the state function code, input size, and a list of sparse availability flags.
        This method searches for a Python file containing the specified function name in the current working directory.
        It extracts the function code using a custom FunctionExtractor class and converts it to C++ code using
        a FunctionToCppVisitor class. It also retrieves the input size from the function code.
        If the function is not found, it raises a FileNotFoundError.
        If the function code cannot be parsed, it raises a SyntaxError.
        """
        function_file_path = ControlDeploy.find_file(
            f"{function_name}.py", os.getcwd())
        state_function_U_size = ExpressionDeploy.get_input_size_from_function_code(
            function_file_path)

        extractor = FunctionExtractor(function_file_path)
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

    @staticmethod
    def generate_EKF_cpp_code(
            ekf: ExtendedKalmanFilter,
            file_name: str = None, number_of_delay: int = 0):
        """
        Generates C++ header files for deploying an Extended Kalman Filter (EKF) based on the provided Python EKF object.
        This function inspects the EKF object, extracts its parameters, state and measurement functions, and their Jacobians,
        and generates corresponding C++ code files for use in a C++ project. The generated files include matrix definitions,
        parameter classes, state and measurement functions, and a factory function to instantiate the EKF in C++.
        Args:
            ekf (ExtendedKalmanFilter): The EKF object containing system matrices, functions, and parameters.
            file_name (str, optional): The base name for the generated files. If None, uses the caller's file name.
            number_of_delay (int, optional): The number of delay steps to include in the EKF type definition.
        Returns:
            List[str]: A list of file names (with extensions) for all generated C++ header files.
        Raises:
            ValueError: If the EKF object or its required attributes are invalid or missing.
        Notes:
            - The generated files are written to disk and include matrix, parameter, function, and EKF type definitions.
            - The function relies on several helper classes and functions (e.g., ControlDeploy, NumpyDeploy, KalmanFilterDeploy).
            - The generated C++ code assumes the existence of certain namespaces and type definitions in the target project.
        """

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
        parameter_code_file_name_without_ext = code_file_name + "_parameter"

        KalmanFilterDeploy.create_cpp_code_file(
            parameter_code_file_name_without_ext, "", parameter_code)

        deployed_file_names.append(
            parameter_code_file_name_without_ext + ".hpp")

        # create state and measurement functions
        fxu_name = ekf.state_function.__module__
        state_function_code_lines, state_function_U_size, _ = \
            KalmanFilterDeploy.create_state_and_measurement_function_code(fxu_name,
                                                                          "X_Type")

        fxu_jacobian_name = ekf.state_function_jacobian.__module__
        state_function_jacobian_code_lines, _, A_SparseAvailable_list = \
            KalmanFilterDeploy.create_state_and_measurement_function_code(fxu_jacobian_name,
                                                                          "A_Type")

        ekf_A = A_SparseAvailable_list[0]

        hx_name = ekf.measurement_function.__module__
        measurement_function_code_lines, _, _ = \
            KalmanFilterDeploy.create_state_and_measurement_function_code(hx_name,
                                                                          "Y_Type")

        hx_jacobian_name = ekf.measurement_function_jacobian.__module__
        measurement_function_jacobian_code_lines, _, C_SparseAvailable_list = \
            KalmanFilterDeploy.create_state_and_measurement_function_code(hx_jacobian_name,
                                                                          "C_Type")

        ekf_C = C_SparseAvailable_list[0]

        # create A, C matrices
        exec(f"{variable_name}_A = ekf_A")
        A_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_A, " +
            "file_name=caller_file_name_without_ext)")
        exec(f"{variable_name}_C = ekf_C")
        C_file_name = eval(
            f"NumpyDeploy.generate_matrix_cpp_code(matrix_in={variable_name}_C, " +
            "file_name=caller_file_name_without_ext)")

        deployed_file_names.append(A_file_name)
        deployed_file_names.append(C_file_name)

        A_file_name_no_extension = A_file_name.split(".")[0]
        C_file_name_no_extension = C_file_name.split(".")[0]

        # generate state function
        state_function_code_suffix = ""
        state_function_code_suffix += f"#include \"{A_file_name}\"\n"
        state_function_code_suffix += f"#include \"{parameter_code_file_name_without_ext}.hpp\"\n\n"

        state_function_code_suffix += "#include \"python_control.hpp\"\n\n"

        state_function_code = ""

        state_function_code += "using namespace PythonControl;\n\n"

        state_function_code += "using Parameter_Type = " + \
            parameter_code_file_name_without_ext + "::Parameter_Type;\n\n"

        state_function_code += "using namespace PythonMath;\n\n"

        state_function_code += f"using A_Type = {A_file_name_no_extension}::type;\n"
        state_function_code += "using X_Type = StateSpaceState_Type<double, A_Type::COLS>;\n"
        state_function_code += f"using U_Type = StateSpaceInput_Type<double, {state_function_U_size}>;\n\n"

        for i, line in enumerate(state_function_code_lines):
            state_function_code += line + "\n"

        state_function_code_file_name_without_ext = code_file_name + "_state_function"

        KalmanFilterDeploy.create_cpp_code_file(
            state_function_code_file_name_without_ext,
            state_function_code_suffix, state_function_code)

        deployed_file_names.append(
            state_function_code_file_name_without_ext + ".hpp")

        # generate state function jacobian
        state_function_jacobian_code_suffix = ""
        state_function_jacobian_code_suffix += f"#include \"{A_file_name}\"\n"
        state_function_jacobian_code_suffix += f"#include \"{C_file_name}\"\n"
        state_function_jacobian_code_suffix += f"#include \"{parameter_code_file_name_without_ext}.hpp\"\n\n"

        state_function_jacobian_code_suffix += "#include \"python_control.hpp\"\n\n"

        state_function_jacobian_code = ""

        state_function_jacobian_code += "using namespace PythonControl;\n\n"

        state_function_jacobian_code += "using Parameter_Type = " + \
            parameter_code_file_name_without_ext + "::Parameter_Type;\n\n"

        state_function_jacobian_code += "using namespace PythonMath;\n\n"

        state_function_jacobian_code += f"using A_Type = {A_file_name_no_extension}::type;\n"
        state_function_jacobian_code += "using X_Type = StateSpaceState_Type<double, A_Type::COLS>;\n"
        state_function_jacobian_code += f"using U_Type = StateSpaceInput_Type<double, {state_function_U_size}>;\n\n"

        for i, line in enumerate(state_function_jacobian_code_lines):
            state_function_jacobian_code += line + "\n"

        state_function_jacobian_code_file_name_without_ext = code_file_name + \
            "_state_function_jacobian"

        KalmanFilterDeploy.create_cpp_code_file(
            state_function_jacobian_code_file_name_without_ext,
            state_function_jacobian_code_suffix,
            state_function_jacobian_code)

        deployed_file_names.append(
            state_function_jacobian_code_file_name_without_ext + ".hpp")

        # generate measurement function
        measurement_function_code_suffix = ""
        measurement_function_code_suffix += f"#include \"{A_file_name}\"\n"
        measurement_function_code_suffix += f"#include \"{C_file_name}\"\n"
        measurement_function_code_suffix += f"#include \"{parameter_code_file_name_without_ext}.hpp\"\n\n"

        measurement_function_code_suffix += "#include \"python_control.hpp\"\n\n"

        measurement_function_code = ""

        measurement_function_code += "using namespace PythonControl;\n\n"

        measurement_function_code += "using Parameter_Type = " + \
            parameter_code_file_name_without_ext + "::Parameter_Type;\n\n"

        measurement_function_code += "using namespace PythonMath;\n\n"

        measurement_function_code += f"using A_Type = {A_file_name_no_extension}::type;\n"
        measurement_function_code += f"using C_Type = {C_file_name_no_extension}::type;\n"
        measurement_function_code += "using X_Type = StateSpaceState_Type<double, A_Type::COLS>;\n"
        measurement_function_code += "using Y_Type = StateSpaceOutput_Type<double, C_Type::COLS>;\n\n"

        for i, line in enumerate(measurement_function_code_lines):
            measurement_function_code += line + "\n"

        measurement_function_code_file_name_without_ext = code_file_name + \
            "_measurement_function"

        KalmanFilterDeploy.create_cpp_code_file(
            measurement_function_code_file_name_without_ext,
            measurement_function_code_suffix,
            measurement_function_code)

        deployed_file_names.append(
            measurement_function_code_file_name_without_ext + ".hpp")

        # generate measurement function jacobian
        measurement_function_jacobian_code_suffix = ""
        measurement_function_jacobian_code_suffix += f"#include \"{A_file_name}\"\n"
        measurement_function_jacobian_code_suffix += f"#include \"{C_file_name}\"\n"
        measurement_function_jacobian_code_suffix += f"#include \"{parameter_code_file_name_without_ext}.hpp\"\n"

        measurement_function_jacobian_code_suffix += "#include \"python_control.hpp\"\n\n"

        measurement_function_jacobian_code = ""

        measurement_function_jacobian_code += "using namespace PythonControl;\n\n"

        measurement_function_jacobian_code += "using Parameter_Type = " + \
            parameter_code_file_name_without_ext + "::Parameter_Type;\n\n"

        measurement_function_jacobian_code += "using namespace PythonMath;\n\n"

        measurement_function_jacobian_code += f"using A_Type = {A_file_name_no_extension}::type;\n"
        measurement_function_jacobian_code += f"using C_Type = {C_file_name_no_extension}::type;\n"
        measurement_function_jacobian_code += "using X_Type = StateSpaceState_Type<double, A_Type::COLS>;\n"
        measurement_function_jacobian_code += "using Y_Type = StateSpaceOutput_Type<double, C_Type::COLS>;\n\n"

        for i, line in enumerate(measurement_function_jacobian_code_lines):
            measurement_function_jacobian_code += line + "\n"

        measurement_function_jacobian_code_file_name_without_ext = code_file_name + \
            "_measurement_function_jacobian"

        KalmanFilterDeploy.create_cpp_code_file(
            measurement_function_jacobian_code_file_name_without_ext,
            measurement_function_jacobian_code_suffix,
            measurement_function_jacobian_code)

        deployed_file_names.append(
            measurement_function_jacobian_code_file_name_without_ext + ".hpp")

        # create cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{A_file_name}\"\n"
        code_text += f"#include \"{C_file_name}\"\n"
        code_text += f"#include \"{parameter_code_file_name_without_ext}.hpp\"\n"
        code_text += f"#include \"{state_function_code_file_name_without_ext}.hpp\"\n"
        code_text += f"#include \"{state_function_jacobian_code_file_name_without_ext}.hpp\"\n"
        code_text += f"#include \"{measurement_function_code_file_name_without_ext}.hpp\"\n"
        code_text += f"#include \"{measurement_function_jacobian_code_file_name_without_ext}.hpp\"\n\n"

        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n\n"

        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {number_of_delay};\n\n"

        code_text += f"using A_Type = {A_file_name_no_extension}::type;\n\n"

        code_text += f"using C_Type = {C_file_name_no_extension}::type;\n\n"

        code_text += "constexpr std::size_t STATE_SIZE = A_Type::COLS;\n"
        code_text += f"constexpr std::size_t INPUT_SIZE = {state_function_U_size};\n"
        code_text += "constexpr std::size_t OUTPUT_SIZE = C_Type::COLS;\n\n"

        code_text += "using X_Type = StateSpaceState_Type<double, STATE_SIZE>;\n"
        code_text += "using U_Type = StateSpaceInput_Type<double, INPUT_SIZE>;\n"
        code_text += "using Y_Type = StateSpaceOutput_Type<double, OUTPUT_SIZE>;\n\n"

        code_text += f"using Q_Type = KalmanFilter_Q_Type<{type_name}, STATE_SIZE>;\n\n"

        code_text += f"using R_Type = KalmanFilter_R_Type<{type_name}, OUTPUT_SIZE>;\n\n"

        code_text += "using Parameter_Type = " + \
            parameter_code_file_name_without_ext + "::Parameter_Type;\n\n"

        code_text += "using type = ExtendedKalmanFilter_Type<\n" + \
            "    A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += "  auto Q = make_KalmanFilter_Q<STATE_SIZE>(\n"
        for i in range(ekf.Q.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(ekf.Q[i, i]) + ")"
            if i == ekf.Q.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += "  );\n\n"

        code_text += "  auto R = make_KalmanFilter_R<OUTPUT_SIZE>(\n"
        for i in range(ekf.R.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(ekf.R[i, i]) + ")"
            if i == ekf.R.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += "  );\n\n"

        code_text += "  Parameter_Type parameters;\n\n"

        code_text += "  StateFunction_Object<X_Type, U_Type, Parameter_Type> state_function_object =\n" + \
            "    [](const X_Type& X, const U_Type& U, const Parameter_Type& Parameters) {\n" + \
            f"      return {state_function_code_file_name_without_ext}::function(X, U, Parameters);\n"
        code_text += "    };\n\n"

        code_text += "  StateFunctionJacobian_Object<A_Type, X_Type, U_Type, Parameter_Type> " + \
            "state_function_jacobian_object =\n" + \
            "    [](const X_Type& X, const U_Type& U, const Parameter_Type& Parameters) {\n" + \
            f"      return {state_function_jacobian_code_file_name_without_ext}::function(X, U, Parameters);\n"
        code_text += "    };\n\n"

        code_text += "  MeasurementFunction_Object<Y_Type, X_Type, Parameter_Type> measurement_function_object =\n" + \
            "    [](const X_Type& X, const Parameter_Type& Parameters) {\n" + \
            f"      return {measurement_function_code_file_name_without_ext}::function(X, Parameters);\n"
        code_text += "    };\n\n"

        code_text += "  MeasurementFunctionJacobian_Object<C_Type, X_Type, Parameter_Type> " + \
            "measurement_function_jacobian_object =\n" + \
            "    [](const X_Type& X, const Parameter_Type& Parameters) {\n" + \
            f"      return {measurement_function_jacobian_code_file_name_without_ext}::function(X, Parameters);\n"
        code_text += "    };\n\n"

        code_text += "  return ExtendedKalmanFilter_Type<\n" + \
            "    A_Type, C_Type, U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>(\n" + \
            "    Q, R, state_function_object, state_function_jacobian_object,\n" + \
            "    measurement_function_object, measurement_function_jacobian_object,\n" + \
            "    parameters);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names

    @staticmethod
    def generate_UKF_cpp_code(ukf: UnscentedKalmanFilter,
                              file_name: str = None, number_of_delay: int = 0):
        """
        Generates C++ code for an Unscented Kalman Filter (UKF) based on the provided UKF object.
        Args:
            ukf (UnscentedKalmanFilter): An instance of the UnscentedKalmanFilter class containing the filter parameters.
            file_name (str, optional): The name of the file to which the generated code will be written. Defaults to None.
            number_of_delay (int, optional): The number of delays to consider in the filter. Defaults to 0.
        Returns:
            list: A list of file names where the generated C++ code is written.
        This method generates C++ code for an Unscented Kalman Filter by inspecting the provided UKF object.
        It restricts the data type of the filter, checks the data type of the Q matrix, and generates C++ code
        for the state space representation of the filter. It also creates a C++ header file containing the
        necessary includes, namespace definitions, and type definitions for the filter.
        The generated code includes the initialization of the filter's state space, covariance matrices, and
        the Kalman gain. It also handles the initialization of the filter's parameters if they are provided.
        The generated C++ code is written to a file named `<caller_file_name_without_ext>_<variable_name>.hpp`,
        where `<caller_file_name_without_ext>` is the name of the file from which the function is called,
        and `<variable_name>` is the name of the UKF variable in the caller's local scope.
        Raises:
            ValueError: If the data type of the Q matrix is not supported.
        """
        deployed_file_names = []

        ControlDeploy.restrict_data_type(ukf.Q.dtype.name)

        type_name = NumpyDeploy.check_dtype(ukf.Q)

        # %% inspect arguments
        # Get the caller's frame
        frame = inspect.currentframe().f_back
        # Get the caller's local variables
        caller_locals = frame.f_locals
        # Find the variable name that matches the matrix_in value
        variable_name = None
        for name, value in caller_locals.items():
            if value is ukf:
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
            ukf.Parameters, type_name)
        parameter_code_file_name_without_ext = code_file_name + "_parameter"

        KalmanFilterDeploy.create_cpp_code_file(
            parameter_code_file_name_without_ext, "", parameter_code)

        deployed_file_names.append(
            parameter_code_file_name_without_ext + ".hpp")

        # create state and measurement functions
        fxu_name = ukf.state_function.__module__
        state_function_code_lines, state_function_U_size, _ = \
            KalmanFilterDeploy.create_state_and_measurement_function_code(fxu_name,
                                                                          "X_Type")

        hx_name = ukf.measurement_function.__module__
        measurement_function_code_lines, _, _ = \
            KalmanFilterDeploy.create_state_and_measurement_function_code(hx_name,
                                                                          "Y_Type")

        # generate state function
        state_size = ukf.Q.shape[0]

        state_function_code_suffix = ""
        state_function_code_suffix += f"#include \"{parameter_code_file_name_without_ext}.hpp\"\n\n"

        state_function_code_suffix += "#include \"python_control.hpp\"\n\n"

        state_function_code_suffix += "using namespace PythonControl;\n\n"

        state_function_code_suffix += "using Parameter_Type = " + \
            parameter_code_file_name_without_ext + "::Parameter_Type;\n\n"

        state_function_code_suffix += "using namespace PythonMath;\n\n"

        state_function_code = ""
        state_function_code += f"using X_Type = StateSpaceState_Type<double, {state_size}>;\n"
        state_function_code += f"using U_Type = StateSpaceInput_Type<double, {state_function_U_size}>;\n\n"

        for i, line in enumerate(state_function_code_lines):
            state_function_code += line + "\n"

        state_function_code_file_name_without_ext = code_file_name + "_state_function"

        KalmanFilterDeploy.create_cpp_code_file(
            state_function_code_file_name_without_ext,
            state_function_code_suffix, state_function_code)

        deployed_file_names.append(
            state_function_code_file_name_without_ext + ".hpp")

        measurement_size = ukf.R.shape[0]

        # generate measurement function
        measurement_function_code_suffix = ""
        measurement_function_code_suffix += f"#include \"{parameter_code_file_name_without_ext}.hpp\"\n\n"

        measurement_function_code_suffix += "#include \"python_control.hpp\"\n\n"

        measurement_function_code_suffix += "using namespace PythonControl;\n\n"

        measurement_function_code_suffix += "using Parameter_Type = " + \
            parameter_code_file_name_without_ext + "::Parameter_Type;\n\n"

        measurement_function_code_suffix += "using namespace PythonMath;\n\n"

        measurement_function_code = ""
        measurement_function_code += f"using X_Type = StateSpaceState_Type<double, {state_size}>;\n"
        measurement_function_code += f"using U_Type = StateSpaceInput_Type<double, {state_function_U_size}>;\n"
        measurement_function_code += f"using Y_Type = StateSpaceOutput_Type<double, {measurement_size}>;\n\n"

        for i, line in enumerate(measurement_function_code_lines):
            measurement_function_code += line + "\n"

        measurement_function_code_file_name_without_ext = code_file_name + \
            "_measurement_function"

        KalmanFilterDeploy.create_cpp_code_file(
            measurement_function_code_file_name_without_ext,
            measurement_function_code_suffix,
            measurement_function_code)

        deployed_file_names.append(
            measurement_function_code_file_name_without_ext + ".hpp")

        # create cpp code
        code_text = ""

        file_header_macro_name = "__" + code_file_name.upper() + "_HPP__"

        code_text += "#ifndef " + file_header_macro_name + "\n"
        code_text += "#define " + file_header_macro_name + "\n\n"

        code_text += f"#include \"{parameter_code_file_name_without_ext}.hpp\"\n"
        code_text += f"#include \"{state_function_code_file_name_without_ext}.hpp\"\n"
        code_text += f"#include \"{measurement_function_code_file_name_without_ext}.hpp\"\n\n"

        code_text += "#include \"python_control.hpp\"\n\n"

        namespace_name = code_file_name

        code_text += "namespace " + namespace_name + " {\n\n"

        code_text += "using namespace PythonNumpy;\n"
        code_text += "using namespace PythonControl;\n\n"

        code_text += f"constexpr std::size_t NUMBER_OF_DELAY = {number_of_delay};\n\n"

        code_text += f"constexpr std::size_t STATE_SIZE = {state_size};\n"
        code_text += f"constexpr std::size_t INPUT_SIZE = {state_function_U_size};\n"
        code_text += f"constexpr std::size_t OUTPUT_SIZE = {measurement_size};\n\n"

        code_text += "using X_Type = StateSpaceState_Type<double, STATE_SIZE>;\n"
        code_text += "using U_Type = StateSpaceInput_Type<double, INPUT_SIZE>;\n"
        code_text += "using Y_Type = StateSpaceOutput_Type<double, OUTPUT_SIZE>;\n\n"

        code_text += f"using Q_Type = KalmanFilter_Q_Type<{type_name}, STATE_SIZE>;\n\n"

        code_text += f"using R_Type = KalmanFilter_R_Type<{type_name}, OUTPUT_SIZE>;\n\n"

        code_text += "using Parameter_Type = " + \
            parameter_code_file_name_without_ext + "::Parameter_Type;\n\n"

        code_text += "using type = UnscentedKalmanFilter_Type<\n" + \
            "    U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>;\n\n"

        code_text += "inline auto make() -> type {\n\n"

        code_text += "  auto Q = make_KalmanFilter_Q<STATE_SIZE>(\n"
        for i in range(ukf.Q.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(ukf.Q[i, i]) + ")"
            if i == ukf.Q.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += "  );\n\n"

        code_text += "  auto R = make_KalmanFilter_R<OUTPUT_SIZE>(\n"
        for i in range(ukf.R.shape[0]):
            code_text += "    static_cast<" + \
                type_name + ">(" + str(ukf.R[i, i]) + ")"
            if i == ukf.R.shape[0] - 1:
                code_text += "\n"
                break
            else:
                code_text += ",\n"
        code_text += "  );\n\n"

        code_text += "  Parameter_Type parameters;\n\n"

        code_text += "  StateFunction_Object<X_Type, U_Type, Parameter_Type> state_function_object =\n" + \
            "    [](const X_Type& X, const U_Type& U, const Parameter_Type& Parameters) {\n" + \
            f"      return {state_function_code_file_name_without_ext}::function(X, U, Parameters);\n"
        code_text += "    };\n\n"

        code_text += "  MeasurementFunction_Object<Y_Type, X_Type, Parameter_Type> measurement_function_object =\n" + \
            "    [](const X_Type& X, const Parameter_Type& Parameters) {\n" + \
            f"      return {measurement_function_code_file_name_without_ext}::function(X, Parameters);\n"
        code_text += "    };\n\n"

        code_text += "  return UnscentedKalmanFilter_Type<\n" + \
            "    U_Type, Q_Type, R_Type, Parameter_Type, NUMBER_OF_DELAY>(\n" + \
            "    Q, R, state_function_object, measurement_function_object,\n" + \
            "    parameters);\n\n"

        code_text += "}\n\n"

        code_text += "} // namespace " + namespace_name + "\n\n"

        code_text += "#endif // " + file_header_macro_name + "\n"

        code_file_name_ext = ControlDeploy.write_to_file(
            code_text, code_file_name_ext)

        deployed_file_names.append(code_file_name_ext)

        return deployed_file_names
