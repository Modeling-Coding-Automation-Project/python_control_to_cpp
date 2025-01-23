#include <iostream>

#include "python_control.hpp"

using namespace PythonNumpy;
using namespace PythonControl;

int main(void) {
  /* Define State Space */
  auto A = make_SparseMatrix<
      SparseAvailable<ColumnAvailable<true, true, false, false>,
                      ColumnAvailable<true, true, true, false>,
                      ColumnAvailable<false, false, true, true>,
                      ColumnAvailable<true, false, true, true>>>(
      1.0, 0.01, -0.51207708, 0.99, 0.02560385, 1.0, 0.01, 1.28019901,
      -0.06400995, 0.898);

  auto B = make_SparseMatrix<
      SparseAvailable<ColumnAvailable<false>, ColumnAvailable<false>,
                      ColumnAvailable<false>, ColumnAvailable<true>>>(0.01);

  auto C = make_SparseMatrix<
      SparseAvailable<ColumnAvailable<true, false, false, false>,
                      ColumnAvailable<true, false, true, false>>>(
      1.0, 1280.19900633784, -64.009950316892);

  auto D = make_SparseMatrixEmpty<double, 2, 1>();

  double dt = 0.01;

  /* If you want to set delay in the state space, insert template argument */
  constexpr std::size_t DELAY_STEP = 2;

  auto sys = make_DiscreteStateSpace<DELAY_STEP>(A, B, C, D, dt);

  /* State Space Simulation */
  for (std::size_t sim_step = 0; sim_step < 100; ++sim_step) {
    auto u = make_StateSpaceInput<1>(1.0);

    sys.update(u);

    std::cout << "Y_0: " << sys.get_Y()(0, 0) << ", ";
    std::cout << "Y_1: " << sys.get_Y()(1, 0) << ", ";
    std::cout << std::endl;
  }

  return 0;
}
