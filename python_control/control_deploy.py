"""
This module provides the ControlDeploy class, which offers utility methods for data type validation and writing code to files.
It is designed to support deployment processes where ensuring correct data types and exporting code are necessary steps.
"""
import os
import sys
sys.path.append(os.getcwd())

import inspect

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy


class ControlDeploy:
    def __init__(self):
        pass

    @staticmethod
    def restrict_data_type(type_name):
        """
        Restricts the allowed data types to 'float32' or 'float64'.

        Args:
            type_name (str): The name of the data type to check.

        Raises:
            ValueError: If the provided data type is not 'float32' or 'float64'.
        """
        flag = False
        if type_name == 'float64':
            flag = True
        elif type_name == 'float32':
            flag = True
        else:
            flag = False

        if not flag:
            raise ValueError(
                "Data type not supported. Please use float32 or float64")

    @staticmethod
    def write_to_file(code_text, code_file_name_ext):
        """
        Writes the provided code text to a file with the specified name and extension.
        Args:
            code_text (str): The code text to write to the file.
            code_file_name_ext (str): The name of the file including its extension.
        Returns:
            str: The full path to the written file.
        Raises:
            ValueError: If the code text is empty or the file name is invalid.
        """
        with open(code_file_name_ext, "w", encoding="utf-8") as f:
            f.write(code_text)

        return code_file_name_ext
