/**
 * @file discrete_pid_controller.cpp
 * @brief Example simulation of a discrete-time PID controller applied to a
 * plant model.
 *
 * This file demonstrates the creation and simulation of a discrete-time PID
 * controller regulating a simple plant, both represented as discrete transfer
 * functions. The simulation runs for 100 time steps, printing the controller
 * output and plant response at each step.
 */
#include <iostream>

#include "python_control.hpp"

using namespace PythonControl;

int main(void) {
  /* Create plant model */
  auto numerator_plant = make_TransferFunctionNumerator<2>(0.015479737715070607,
                                                           0.01497228851342225);

  auto denominator_plant = make_TransferFunctionDenominator<3>(
      1.0, -1.9048374180359595, 0.9048374180359595);

  double dt = 0.2;

  auto plant =
      make_DiscreteTransferFunction(numerator_plant, denominator_plant, dt);

  /* Create controller model */
  double Kp = 1.0;
  double Ki = 0.1;
  double Kd = 0.5;
  double N = 1.0 / dt;
  double Kb = Ki;
  double minimum_output = -0.2;
  double maximum_output = 0.2;

  auto pid_controller = make_DiscretePID_Controller(
      dt, Kp, Ki, Kd, N, Kb, minimum_output, maximum_output);

  /* Simulation */
  double reference = 1.0;
  double y = 0.0;

  for (std::size_t i = 0; i < 100; i++) {
    double error = reference - y;

    double u = pid_controller.update(error);

    plant.update(u);

    y = plant.get_y();

    std::cout << "u: " << u << ", ";
    std::cout << "y: " << y << std::endl;
  }

  return 0;
}
