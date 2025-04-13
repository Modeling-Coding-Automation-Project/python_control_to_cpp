import os
import sys
sys.path.append(os.getcwd())

import control
import numpy as np
import matplotlib.pyplot as plt

from python_control.pid_controller import DiscretePID_Controller
from python_control.pid_controller_deploy import DiscretePID_ControllerDeploy
from test_sil.SIL_operator import SIL_CodeGenerator
from test_vs.MCAP_tester.tester.MCAP_tester import MCAPTester

# parameter
dt = 0.2
time_series = np.arange(0, 20, dt)

Kp = 1.0
Ki = 0.1
Kd = 0.5

plant_model = control.TransferFunction([1.6], [2.0, 1.0, 0.0])
plant_model_d = plant_model.sample(Ts=dt, method='zoh')
plant_model_d_ss = control.ss(plant_model_d)
x_plant = np.zeros((plant_model_d_ss.A.shape[0], 1))

pid = DiscretePID_Controller(delta_time=dt, Kp=Kp, Ki=Ki, Kd=Kd, N=(
    1.0 / dt), Kb=Ki, minimum_output=-0.2, maximum_output=0.2)

deployed_file_names = DiscretePID_ControllerDeploy.generate_PID_cpp_code(
    pid)
current_dir = os.path.dirname(__file__)
generator = SIL_CodeGenerator(deployed_file_names, current_dir)
generator.build_SIL_code()

from test_sil.pid_controller import PidControllerSIL
PidControllerSIL.initialize()

# simulation
tester = MCAPTester()
NEAR_LIMIT = 1e-5

r = 1.0
y = 0.0
for i in range(len(time_series)):
    e = r - y

    # controller
    u = pid.update(e)

    if isinstance(e, np.ndarray):
        e = float(e[0, 0])
    u_cpp = PidControllerSIL.update(e)

    # plant
    y = plant_model_d_ss.C @ x_plant + plant_model_d_ss.D * u
    x_plant = plant_model_d_ss.A @ x_plant + plant_model_d_ss.B * u

    # test
    if isinstance(u, np.ndarray):
        u = float(u[0, 0])
    tester.expect_near(u_cpp, u, NEAR_LIMIT,
                       "Discrete PID controller, check u.")
