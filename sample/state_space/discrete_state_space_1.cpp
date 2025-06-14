/**
 * @file discrete_state_space_1.cpp
 * @brief Example simulation of a discrete-time state-space system in C++.
 *
 * This file demonstrates the creation and simulation of a discrete-time
 * state-space model. The state-space system is defined by matrices A, B, C, and
 * D, and is simulated over 50 time steps with a constant input. The state and
 * output values are printed at each simulation step.
 */
#include <iostream>

#include "python_control.hpp"
#include "python_numpy.hpp"

using namespace PythonNumpy;
using namespace PythonControl;

int main(void) {
  /* Define State Space */
  auto A = make_DenseMatrix<2, 2>(0.7, 0.2, -0.3, 0.8);
  auto B = make_DenseMatrix<2, 1>(0.1, 0.2);
  auto C = make_DenseMatrix<1, 2>(2.0, 0.0);
  auto D = make_DenseMatrix<1, 1>(0.0);
  double dt = 0.01;

  auto sys = make_DiscreteStateSpace(A, B, C, D, dt);

  /* State Space Simulation */
  for (std::size_t sim_step = 0; sim_step < 50; ++sim_step) {
    auto u = make_StateSpaceInput<1>(1.0);

    sys.update(u);

    std::cout << "X_0: " << sys.get_X()(0, 0) << ", ";
    std::cout << "X_1: " << sys.get_X()(1, 0) << ", ";
    std::cout << "Y: " << sys.get_Y()(0, 0) << ", ";
    std::cout << std::endl;
  }

  return 0;
}
