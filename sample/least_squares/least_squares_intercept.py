"""Compare intercept modes, generate C++ factories, and plot both fitted lines.

Run with --no-plot for automated checks. Generated headers are written to the
repository root, independently of the current working directory.
"""
import argparse
import os
from pathlib import Path
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY))

from external_libraries.MCAP_python_control.python_control.least_squares import LeastSquares
from python_control.least_squares_deploy import LeastSquaresDeploy
from sample.simulation_manager.visualize.simulation_plotter_dash import SimulationPlotterDash


def run_sample():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([[2.0], [3.0], [4.0]])
    with_intercept = LeastSquares(X)  # fit_intercept=True is the default.
    without_intercept = LeastSquares(X, fit_intercept=False)
    with_intercept.fit(X, y)
    without_intercept.fit(X, y)
    np.testing.assert_allclose(with_intercept.get_weights(), [[1.0], [1.0]])
    np.testing.assert_allclose(without_intercept.get_weights(), [[10.0 / 7.0]])
    np.testing.assert_allclose(
        without_intercept.predict(np.zeros_like(X)), 0.0)
    # Dropping the affine model's last weight does not solve the same problem.
    assert not np.allclose(with_intercept.get_weights()[:-1],
                           without_intercept.get_weights())

    previous_directory = Path.cwd()
    try:
        os.chdir(REPOSITORY)
        # These factories preserve the intercept mode; they create untrained
        # models. The paired C++ example fits the same data on the C++ side.
        deployed_file_names = []
        for ls in (with_intercept, without_intercept):
            deployed_file_names += LeastSquaresDeploy.generate_LS_cpp_code(ls)
    finally:
        os.chdir(previous_directory)

    print(deployed_file_names)
    for model in (with_intercept, without_intercept):
        print(f"fit_intercept={model.fit_intercept}: "
              f"weights={model.get_weights().ravel()}")
    return X, y, with_intercept, without_intercept


def plot_sample(X, y, with_intercept, without_intercept):

    plotter = SimulationPlotterDash(activate_dump=False)
    signals = {"target": y,
               "with intercept": with_intercept.predict(X),
               "without intercept": without_intercept.predict(X)}
    for name, signal in signals.items():
        plotter.append_sequence_name(signal, name)
        plotter.assign(name, position=(0, 0), x_sequence=X, label=name)
    plotter.plot("Least squares: intercept selection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    result = run_sample()
    if not args.no_plot:
        plot_sample(*result)
