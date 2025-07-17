import os
import sys
sys.path.append(os.getcwd())

import math
import numpy as np
import sympy as sp
from dataclasses import dataclass

from external_libraries.MCAP_python_control.python_control.kalman_filter import ExtendedKalmanFilter
from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy
from python_control.kalman_filter_deploy import KalmanFilterDeploy

from sample.simulation_manager.signal_edit.sampler import PulseGenerator

from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester


def create_model(delta_time: float):
    # define parameters and variables
    m, u, v, r, F_f, F_r = sp.symbols('m u v r F_f F_r', real=True)
    I, l_f, l_r, v_dot, r_dot, V, beta, beta_dot = sp.symbols(
        'I l_f l_r v_dot r_dot V beta beta_dot', real=True)

    # derive equations of two wheel vehicle model
    eq_1 = sp.Eq(m * (v_dot + u * r), F_f + F_r)
    eq_2 = sp.Eq(I * r_dot, l_f * F_f - l_r * F_r)

    lhs = eq_1.lhs.subs({u: V, v_dot: V * beta_dot})
    eq1 = sp.Eq(lhs, eq_1.rhs)

    K_f, K_r, delta, beta_f, beta_r = sp.symbols(
        'K_f K_r delta beta_f beta_r', real=True)

    rhs = eq1.rhs.subs({F_f: -2 * K_f * beta_f, F_r: -2 * K_r * beta_r})
    eq_1 = sp.Eq(eq1.lhs, rhs)

    rhs = eq_2.rhs.subs({F_f: -2 * K_f * beta_f, F_r: -2 * K_r * beta_r})
    eq_2 = sp.Eq(eq_2.lhs, rhs)

    rhs = eq_1.rhs.subs({
        beta_f: beta + (l_f / V) * r - delta,
        beta_r: beta - (l_r / V) * r
    })
    eq_1 = sp.Eq(eq_1.lhs, rhs)

    rhs = eq_2.rhs.subs({
        beta_f: beta + (l_f / V) * r - delta,
        beta_r: beta - (l_r / V) * r
    })
    eq_2 = sp.Eq(eq_2.lhs, rhs)

    eq_vec = [eq_1, eq_2]

    solution = sp.solve(eq_vec, beta_dot, dict=True)
    beta_dot_sol = sp.simplify(solution[0][beta_dot])

    solution = sp.solve(eq_vec, r_dot, dict=True)
    r_dot_sol = sp.simplify(solution[0][r_dot])

    # Define state space model
    accel = sp.symbols('accel', real=True)
    U = sp.Matrix([[delta], [accel]])

    theta, px, py = sp.symbols('theta px py', real=True)
    X = sp.Matrix([[px], [py], [theta], [r], [beta], [V]])
    Y = sp.Matrix([[px], [py], [theta], [r], [V]])

    fxu_continuous = sp.Matrix([
        [V * sp.cos(theta)],
        [V * sp.sin(theta)],
        [r],
        [r_dot_sol],
        [beta_dot_sol],
        [accel],
    ])
    fxu: sp.Matrix = X + fxu_continuous * delta_time

    hx = sp.Matrix([[X[0]], [X[1]], [X[2]], [X[3]], [X[5]]])

    # derive Jacobian
    fxu_jacobian = fxu.jacobian(X)
    hx_jacobian = hx.jacobian(X)

    fxu_file_name = ExpressionDeploy.write_state_function_code_from_sympy(
        fxu, X, U)
    fxu_jacobian_file_name = \
        ExpressionDeploy.write_state_function_code_from_sympy(
            fxu_jacobian, X, U)

    hx_file_name = ExpressionDeploy.write_measurement_function_code_from_sympy(
        hx, X)
    hx_jacobian_file_name = \
        ExpressionDeploy.write_measurement_function_code_from_sympy(
            hx_jacobian, X)

    return X, U, Y, \
        fxu_file_name, fxu_jacobian_file_name, \
        hx_file_name, hx_jacobian_file_name


@dataclass
class Parameter:
    m: float = 2000
    l_f: float = 1.4
    l_r: float = 1.6
    I: float = 4000
    K_f: float = 12e3
    K_r: float = 11e3


def main():
    # simulation setup
    sim_delta_time = 0.01
    simulation_time = 20.0

    time = np.arange(0, simulation_time, sim_delta_time)

    X, U, Y, \
        fxu_file_name, fxu_jacobian_file_name, \
        hx_file_name, hx_jacobian_file_name = create_model(sim_delta_time)

    parameters_ekf = Parameter()

    Q_ekf = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    R_ekf = np.diag([1.0, 1.0, 1.0, 1.0, 1.0])

    local_vars = {}

    exec(f"from {fxu_file_name} import function as fxu_script_function",
         globals(), local_vars)
    exec(
        f"from {fxu_jacobian_file_name} import function as fxu_jacobian_script_function", globals(), local_vars)
    exec(f"from {hx_file_name} import function as hx_script_function",
         globals(), local_vars)
    exec(
        f"from {hx_jacobian_file_name} import function as hx_jacobian_script_function", globals(), local_vars)

    fxu_script_function = local_vars["fxu_script_function"]
    fxu_jacobian_script_function = local_vars["fxu_jacobian_script_function"]
    hx_script_function = local_vars["hx_script_function"]
    hx_jacobian_script_function = local_vars["hx_jacobian_script_function"]

    ekf = ExtendedKalmanFilter(
        state_function=fxu_script_function,
        measurement_function=hx_script_function,
        state_function_jacobian=fxu_jacobian_script_function,
        measurement_function_jacobian=hx_jacobian_script_function,
        Q=Q_ekf,
        R=R_ekf,
        Parameters=parameters_ekf
    )

    # You can create cpp header which can easily define EKF as C++ code
    deployed_file_names = KalmanFilterDeploy.generate_EKF_cpp_code(ekf)

    current_dir = os.path.dirname(__file__)
    generator = SIL_CodeGenerator(deployed_file_names, current_dir)
    generator.build_SIL_code()

    from test_sil.kalman_filter import KalmanFilterSIL
    KalmanFilterSIL.initialize()

    # X: px, py, theta, r, beta, V
    x_true = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]])

    ekf.x_hat = np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.5]])

    # U: delta, accel
    u = np.array([[0.0], [0.0]])

    # create delta sequence
    _, input_signal = PulseGenerator.sample_pulse(
        sampling_interval=sim_delta_time,
        start_time=time[0],
        period=6.0,
        pulse_width=50.0,
        pulse_amplitude=2.0,
        duration=time[-1],
    )
    input_signal = input_signal - 1.0
    delta_sequence = np.zeros_like(time)

    delta_max = 30.0 / 180.0 / math.pi
    for i in range(len(time)):
        if i == 0:
            delta_sequence[i] = input_signal[i, 0] * delta_max * sim_delta_time
        else:
            delta_sequence[i] = delta_sequence[i - 1] + \
                input_signal[i, 0] * delta_max * sim_delta_time

    # create sequence for acceleration
    _, signal_plus = PulseGenerator.sample_pulse(
        sampling_interval=sim_delta_time,
        start_time=time[0],
        period=10.0,
        pulse_width=25.0,
        pulse_amplitude=1.0,
        duration=time[-1],
    )

    _, signal_minus = PulseGenerator.sample_pulse(
        sampling_interval=sim_delta_time,
        start_time=time[0] + 5.0,
        period=10.0,
        pulse_width=25.0,
        pulse_amplitude=-1.0,
        duration=time[-1],
    )

    accel_sequence = np.zeros_like(time)
    for i in range(len(time)):
        accel_sequence[i] = signal_plus[i, 0] + signal_minus[i, 0]

    tester = MCAPTester()
    NEAR_LIMIT = 1e-5

    # simulation
    for i in range(round(simulation_time / sim_delta_time)):
        u = np.array([[delta_sequence[i]], [accel_sequence[i]]])

        # system response
        x_true = fxu_script_function(x_true, u, parameters_ekf)
        y_measured = hx_script_function(x_true, parameters_ekf)

        # estimate
        ekf.predict(u)
        ekf.update(y_measured)

        x_estimated = ekf.get_x_hat_without_delay()

        KalmanFilterSIL.predict_and_update(u, y_measured)
        x_estimate_cpp = KalmanFilterSIL.get_x_hat()

        tester.expect_near(x_estimate_cpp, x_estimated, NEAR_LIMIT,
                           "Extended Kalman Filter SIL, check x_hat.")

    tester.throw_error_if_test_failed()


if __name__ == "__main__":
    main()
