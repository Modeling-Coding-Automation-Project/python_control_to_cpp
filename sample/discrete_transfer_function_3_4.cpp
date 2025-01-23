#include <iostream>

#include "python_control.hpp"

using namespace PythonNumpy;
using namespace PythonControl;

int main(void) {
  /* Define transfer function */
  auto numerator_3_4 = make_TransferFunctionNumerator<4>(
      0.0012642614672828678, 0.0037594540384011665, -0.002781625665309928,
      -0.0009364784774175128);

  auto denominator_3_4 = make_TransferFunctionNumerator<5>(
      1.0, -3.565195017021459, 4.815115383504625, -2.9189348011558485,
      0.6703200460356397);

  double dt = 0.2;

  /* If you want to set delay in the state space, insert template argument */
  constexpr std::size_t DELAY_STEP = 2;

  auto system_3_4 = make_DiscreteTransferFunction<DELAY_STEP>(
      numerator_3_4, denominator_3_4, dt);

  /* Transfer Function Simulation */
  for (std::size_t i = 0; i < 70; i++) {
    double u = 1.0;
    system_3_4.update(u);

    std::cout << "y: " << system_3_4.get_y() << ", ";
    std::cout << std::endl;
  }
  std::cout << std::endl;

  /* Derive steady state and input from specified output value. */
  double y_steady_state = 1.0;

  double u_steady_state =
      system_3_4.solve_steady_state_and_input(y_steady_state);

  auto x_steady_state = system_3_4.get_X();

  std::cout << "if y_steady_state is " << y_steady_state << ", " << std::endl;
  std::cout << "u_steady_state: " << u_steady_state << std::endl;
  std::cout << "x_steady_state_0: " << x_steady_state(0, 0) << std::endl;
  std::cout << "x_steady_state_1: " << x_steady_state(1, 0) << std::endl;
  std::cout << "x_steady_state_2: " << x_steady_state(2, 0) << std::endl;
  std::cout << "x_steady_state_3: " << x_steady_state(3, 0) << std::endl;

  return 0;
}
