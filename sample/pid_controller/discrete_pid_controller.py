import os
import sys
sys.path.append(os.getcwd())

import control
import numpy as np
import matplotlib.pyplot as plt

from sample.simulation_manager.visualize.simulation_plotter import SimulationPlotter

# parameter
dt = 0.2
time_series = np.arange(0, 20, dt)

'''
Continuous PID controller
'''
# plant model
plant_model = control.TransferFunction([1.6], [2.0, 1.0, 0.0])

# pid controller
Kp = 1.0
Ki = 0.1
Kd = 0.5
pid_controller = control.TransferFunction([Kd, Kp, Ki], [1.0, 0.0])

# continuous system
system = control.feedback(pid_controller * plant_model, 1, sign=-1)

# step response of system
t, y = control.step_response(system, T=time_series)

# plot results of continuous system
plt.plot(t, y)
plt.title('Step Response of continuous PID controller')
plt.xlabel('Time (s)')
plt.ylabel('Response')
plt.legend(['continuous'])
plt.grid(True)


'''
Discrete PID controller
'''
# discretize plant model
plant_model_d = plant_model.sample(Ts=dt, method='zoh')
print("\nDiscrete plant model:")
print(plant_model_d)

# discretize pid controller
P_discrete = control.TransferFunction([Kp], [1.0], dt)
I_discrete = control.TransferFunction([Ki * dt, 0], [1.0, -1.0], dt)
D_discrete = control.TransferFunction([Kd, -Kd], [dt, 0.0], dt)

pid_controller_d = P_discrete + I_discrete + D_discrete
print("\nDiscrete PID controller:")
print(pid_controller_d)

# discrete system
system_d = control.feedback(pid_controller_d * plant_model_d, 1, sign=-1)

# step response of system
t, y = control.step_response(system_d, T=time_series)

# plot results of discrete system
plt.plot(t, y)
plt.legend(['continuous', 'discrete'])
plt.grid(True)

'''
Discrete PID controller each step simulation
'''
plant_model_d_ss = control.ss(plant_model_d)
x_plant = np.zeros((plant_model_d_ss.A.shape[0], 1))

PID_discrete_ss = control.ss(pid_controller_d)
x_PID_discrete = np.zeros((PID_discrete_ss.A.shape[0], 1))

plotter = SimulationPlotter()

r = 1.0
y = 0.0
for i in range(len(time_series)):
    e = r - y

    # controller
    u = PID_discrete_ss.C @ x_PID_discrete + PID_discrete_ss.D * e
    x_PID_discrete = PID_discrete_ss.A @ x_PID_discrete + PID_discrete_ss.B * e

    # plant
    y = plant_model_d_ss.C @ x_plant + plant_model_d_ss.D * u
    x_plant = plant_model_d_ss.A @ x_plant + plant_model_d_ss.B * u

    plotter.append(y)

plotter.assign("y", position=(0, 0), x_sequence=t)

plotter.plot("Discrete PID controller each step simulation")
