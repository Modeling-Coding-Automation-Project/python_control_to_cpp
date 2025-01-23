#include <iostream>

#include "python_control.hpp"

using namespace PythonNumpy;
using namespace PythonControl;

int main(void) {
  /* Define transfer function */
  auto numerator_2_4 = make_TransferFunctionNumerator<3>(0.5, 0.3, 0.1);

  auto denominator_2_4 =
      make_TransferFunctionNumerator<5>(1.0, -1.8, 1.5, -0.7, 0.2);

  double dt = 0.2;

  auto system_2_4 =
      make_DiscreteTransferFunction(numerator_2_4, denominator_2_4, dt);

  /* Transfer Function Simulation */
  for (std::size_t i = 0; i < 30; i++) {
    double u = 1.0;
    system_2_4.update(u);

    std::cout << "y: " << system_2_4.get_y() << ", ";
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Derive steady state and input from specified output value. */
  double y_steady_state = 1.0;

  double u_steady_state =
      system_2_4.solve_steady_state_and_input(y_steady_state);

  auto x_steady_state = system_2_4.get_X();

  std::cout << "if y_steady_state is " << y_steady_state << ", " << std::endl;
  std::cout << "u_steady_state: " << u_steady_state << std::endl;
  std::cout << "x_steady_state_0: " << x_steady_state(0, 0) << std::endl;
  std::cout << "x_steady_state_1: " << x_steady_state(1, 0) << std::endl;
  std::cout << "x_steady_state_2: " << x_steady_state(2, 0) << std::endl;
  std::cout << "x_steady_state_3: " << x_steady_state(3, 0) << std::endl;

  return 0;
}
