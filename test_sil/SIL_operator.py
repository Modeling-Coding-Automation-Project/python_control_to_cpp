import os
import sys


class SIL_CodeGenerator:
    def __init__(self, deployed_file_names, SIL_folder):
        self.deployed_file_names = deployed_file_names
        self.SIL_folder = SIL_folder

        self.folder_name = os.path.basename(os.path.normpath(self.SIL_folder))

    def move_deployed_files(self):
        for file_name in self.deployed_file_names:
            src = os.path.join(os.getcwd(), file_name)
            dst = os.path.join(self.SIL_folder, file_name)
            if os.path.exists(dst):
                os.remove(dst)
            os.rename(src, dst)

    def generate_wrapper_code(self):

        file_name = f"{self.folder_name}" + "_SIL_wrapper.hpp"
        file_name_without_extension = file_name.split(".")[0]

        deployed_file_name = self.deployed_file_names[-1]
        deployed_file_name_without_extension = deployed_file_name.split(".")[0]

        code_text = ""
        code_text += f"#include \"{deployed_file_name}\"\n\n"

        code_text += f"namespace {file_name_without_extension} = {deployed_file_name_without_extension};\n"

        with open(os.path.join(self.SIL_folder, file_name), "w", encoding="utf-8") as f:
            f.write(code_text)

    def build_SIL_code(self):
        self.move_deployed_files()
        self.generate_wrapper_code()
