"""Generate, compile and compare LS intercept modes against Python/NumPy.

Run from any directory with the repository's Python dependencies installed:
    python test_sil/least_squares_intercept/least_squares_intercept_SIL.py
Build products and generated headers are kept in a temporary directory.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig
import tempfile
from types import SimpleNamespace

import numpy as np
import pybind11

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from external_libraries.MCAP_python_control.python_control.least_squares import LeastSquares
from python_control.least_squares_deploy import LeastSquaresDeploy


def make_cases():
    cases = []
    for dtype in (np.float32, np.float64):
        for features in (1, 2, 3):
            if features == 1:
                x = np.array([[1], [2], [3]], dtype=dtype)
                y = np.array([[2], [3], [4]], dtype=dtype)
            else:
                rng = np.random.default_rng(120 + features)
                x = rng.uniform(-1, 1, (12, features)).astype(dtype)
                coefficients = np.array([1.5, -0.8, 0.4], dtype=dtype)[:features]
                y = (x @ coefficients + dtype(0.35)).reshape(-1, 1)
            for mode in (True, False):
                model = LeastSquares(x, fit_intercept=mode)
                name = f"ls_{dtype.__name__}_{features}_{int(mode)}"
                cases.append((name, model, model, x, y))

        # An object without the new attribute must still generate intercept-on C++.
        x = np.array([[1], [2], [3]], dtype=dtype)
        y = np.array([[2], [3], [4]], dtype=dtype)
        reference = LeastSquares(x)
        legacy = SimpleNamespace(state_size=reference.state_size,
                                 number_of_data=reference.number_of_data,
                                 weights=reference.weights.copy())
        cases.append((f"legacy_{dtype.__name__}", legacy, reference, x, y))
    return cases


def generate_cases(cases, directory):
    includes = []
    bindings = []
    original_directory = Path.cwd()
    try:
        os.chdir(directory)
        for name, model, reference, _, _ in cases:
            [header] = LeastSquaresDeploy.generate_LS_cpp_code(model, file_name=name)
            namespace = Path(header).stem
            expected_mode = "true" if reference.fit_intercept else "false"
            assert f"FIT_INTERCEPT = {expected_mode};" in Path(header).read_text()
            includes.append(f'#include "{header}"')
            bindings.append(
                f'  bind_model<{namespace}::type>(module, "{name}", &{namespace}::make);')
    finally:
        os.chdir(original_directory)
    (directory / "generated_cases.hpp").write_text(
        "\n".join(includes) + "\n\n"
        "inline void bind_generated_cases(py::module_ &module) {\n"
        + "\n".join(bindings) + "\n}\n", encoding="utf-8")


def build_module(directory):
    source = Path(__file__).with_suffix(".cpp")
    output = directory / ("LeastSquaresInterceptSIL" + sysconfig.get_config_var("EXT_SUFFIX"))
    include_paths = [
        directory, REPO / "python_control",
        REPO / "external_libraries/python_numpy_to_cpp/python_numpy",
        REPO / "external_libraries/python_numpy_to_cpp/base_matrix",
        REPO / "external_libraries/python_math_to_cpp/python_math",
        REPO / "external_libraries/python_math_to_cpp/base_math",
        REPO / "external_libraries/base_utility_cpp/base_utility",
        Path(pybind11.get_include()), Path(sysconfig.get_path("include")),
    ]
    command = shlex.split(os.environ.get("CXX", "g++"))
    command += ["-std=c++11", "-O2", "-Wall", "-Wextra", "-Werror", "-shared", "-fPIC"]
    command += [f"-I{path}" for path in include_paths]
    command += [str(source), "-o", str(output)]
    subprocess.run(command, check=True)
    spec = importlib.util.spec_from_file_location("LeastSquaresInterceptSIL", output)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    cases = make_cases()
    with tempfile.TemporaryDirectory(prefix="least_squares_intercept_") as temporary:
        directory = Path(temporary)
        generate_cases(cases, directory)
        module = build_module(directory)
        for name, _, reference, x, y in cases:
            reference.fit(x, y)
            weights, prediction, zero_prediction = getattr(module, name)(x, y)
            expected_shape = (x.shape[1] + int(reference.fit_intercept), 1)
            assert weights.shape == expected_shape, name
            assert weights.dtype == x.dtype, name
            design = (np.column_stack((x, np.ones(x.shape[0], dtype=x.dtype)))
                      if reference.fit_intercept else x)
            expected_weights = np.linalg.lstsq(design, y, rcond=None)[0]
            for actual, expected in (
                    (reference.get_weights(), expected_weights),
                    (weights, expected_weights),
                    (prediction, reference.predict(x)),
                    (zero_prediction, reference.predict(np.zeros_like(x)))):
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-5,
                                           err_msg=name)
            if not reference.fit_intercept:
                np.testing.assert_array_equal(zero_prediction, np.zeros_like(y))
            if x.shape[1] == 1:
                expected = [[1], [1]] if reference.fit_intercept else [[10 / 7]]
                np.testing.assert_allclose(weights, expected, rtol=0, atol=1e-5,
                                           err_msg=name)
            print(f"PASS {name}: weights, predictions, zero input")
    print(f"Least Squares intercept SIL: {len(cases)} cases passed.")


if __name__ == "__main__":
    main()
