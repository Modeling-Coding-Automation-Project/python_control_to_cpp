import os
import sys
sys.path.append(os.getcwd())

import numpy as np
from dataclasses import dataclass

from external_libraries.MCAP_python_control.python_control.pid_controller import DiscretePID_Controller
from python_control.pid_controller_deploy import DiscretePID_ControllerDeploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester


@dataclass
class ServoParams:
    R: float = 2.0           # Omega
    L: float = 5.0e-3        # H
    J: float = 1.0e-2        # kg·m^2
    B: float = 1.0e-3        # N·m·s/rad
    Kt: float = 0.1          # N·m/A
    Kb: float = 0.1          # V·s/rad


def servo_state_matrices(p: ServoParams):
    A = np.array([
        [0, 1, 0],
        [0, -p.B / p.J, p.Kt / p.J],
        [0, -p.Kb / p.L, -p.R / p.L]
    ])
    B = np.array([[0],
                  [0],
                  [1 / p.L]])
    C = np.array([
        [1, 0, 0],   # position θ
        [0, 1, 0]    # speed w
    ])
    D = np.zeros((2, 1))
    return A, B, C, D


if __name__ == "__main__":
    params = ServoParams()
    A, B, C, D = servo_state_matrices(params)

    # euler forward method simulation
    sim_delta_time = 1e-3
    x = np.zeros((3, 1))
    u = np.array([[0.0]])
    simulation_time = 10.0

    # PID controller
    Kp = 2.0
    Ki = 0.5
    Kd = 1.0
    pid = DiscretePID_Controller(delta_time=sim_delta_time, Kp=Kp, Ki=Ki, Kd=Kd, N=(
        0.1 / sim_delta_time), Kb=Ki, minimum_output=-12.0, maximum_output=12.0)

    deployed_file_names = DiscretePID_ControllerDeploy.generate_PID_cpp_code(
        pid)
    current_dir = os.path.dirname(__file__)
    generator = SIL_CodeGenerator(deployed_file_names, current_dir)
    generator.build_SIL_code()

    from test_sil.pid_controller import PidControllerSIL
    PidControllerSIL.initialize()

    theta_ref = np.array([[1.0]])

    time = np.arange(0, simulation_time, sim_delta_time)
    np.random.seed(0)  # for reproducibility

    tester = MCAPTester()
    NEAR_LIMIT = 1e-5

    for k in range(round(simulation_time / sim_delta_time)):
        # plant response
        u = np.array(u).reshape((1, 1))
        u_offset = np.array([[0.1]])

        x_dot = A @ x + B @ (u + u_offset)
        x += sim_delta_time * x_dot           # x(k+1) = x(k) + dt * ẋ

        system_noise = np.random.normal(0, 0.01, size=(C.shape[0], 1))

        y = C @ x + D @ u + system_noise

        # PID controller
        e = theta_ref - y[0, 0]
        u = pid.update(e)

        if isinstance(e, np.ndarray):
            e = float(e[0, 0])
        u_cpp = PidControllerSIL.update(e)

        if isinstance(u, np.ndarray):
            u = float(u[0, 0])
        tester.expect_near(u_cpp, u, NEAR_LIMIT,
                           "servo motor, Discrete PID controller, check u.")

    tester.throw_error_if_test_failed()
