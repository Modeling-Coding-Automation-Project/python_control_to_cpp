/**
 * @file discrete_transfer_function_4_4.cpp
 * @brief Example usage of a discrete transfer function system in C++.
 *
 * This file demonstrates the creation and simulation of a discrete transfer
 * function system with a 4th-order numerator and denominator. The code
 * initializes the transfer function coefficients, simulates the system response
 * to a step input, and computes the steady-state input required to achieve a
 * specified steady-state output.
 */
#include <iostream>

#include "python_control.hpp"

using namespace PythonControl;

int main(void) {
  /* Define transfer function */
  constexpr std::size_t NUMERATOR_SIZE = 5;
  auto numerator_4_4 =
      make_TransferFunctionNumerator<NUMERATOR_SIZE>(1.0, 0.5, 0.3, 0.2, 0.1);

  constexpr std::size_t DENOMINATOR_SIZE = 5;
  auto denominator_4_4 = make_TransferFunctionDenominator<DENOMINATOR_SIZE>(
      1.1, -0.5, 0.4, -0.3, 0.2);

  double dt = 0.2;

  auto system_4_4 =
      make_DiscreteTransferFunction(numerator_4_4, denominator_4_4, dt);

  /* Transfer Function Simulation */
  for (std::size_t i = 0; i < 26; i++) {
    double u = 1.0;
    system_4_4.update(u);

    std::cout << "y: " << system_4_4.get_y() << ", ";
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Derive steady state and input from specified output value. */
  double y_steady_state = 1.0;

  double u_steady_state =
      system_4_4.solve_steady_state_and_input(y_steady_state);

  auto x_steady_state = system_4_4.get_X();

  std::cout << "if y_steady_state is " << y_steady_state << ", " << std::endl;
  std::cout << "u_steady_state: " << u_steady_state << std::endl;
  std::cout << "x_steady_state_0: " << x_steady_state(0, 0) << std::endl;
  std::cout << "x_steady_state_1: " << x_steady_state(1, 0) << std::endl;
  std::cout << "x_steady_state_2: " << x_steady_state(2, 0) << std::endl;
  std::cout << "x_steady_state_3: " << x_steady_state(3, 0) << std::endl;

  return 0;
}
