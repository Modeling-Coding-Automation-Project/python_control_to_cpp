import os
import sys
sys.path.append(os.getcwd())

import inspect

from external_libraries.python_numpy_to_cpp.python_numpy.numpy_deploy import NumpyDeploy


class ControlDeploy:
    def __init__(self):
        pass

    @staticmethod
    def check_data_type(state_space):
        if state_space.A.dtype.name == 'float64':
            return True
        elif state_space.A.dtype.name == 'float32':
            return True
        else:
            return False

    @staticmethod
    def write_to_file(code_text, code_file_name_ext):
        # write to file
        with open(code_file_name_ext, "w", encoding="utf-8") as f:
            f.write(code_text)

        return code_file_name_ext
