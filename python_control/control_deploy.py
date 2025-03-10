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
        # write to file
        with open(code_file_name_ext, "w", encoding="utf-8") as f:
            f.write(code_text)

        return code_file_name_ext
