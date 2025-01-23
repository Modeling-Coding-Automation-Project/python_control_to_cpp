#include <iostream>

#include "python_control.hpp"

using namespace PythonNumpy;
using namespace PythonControl;

int main(void) {
  /* Define State Space */
  using A_Type = Matrix<DefDense, double, 2, 2>;
  using B_Type = Matrix<DefDense, double, 2, 1>;
  using C_Type = Matrix<DefDense, double, 1, 2>;
  using D_Type = Matrix<DefDense, double, 1, 1>;

  auto A = make_DenseMatrix<2, 2>(0.7, 0.2, -0.3, 0.8);
  auto B = make_DenseMatrix<2, 1>(0.1, 0.2);
  auto C = make_DenseMatrix<1, 2>(2, 0);
  auto D = make_DenseMatrix<1, 1>(0);
  double dt = 0.01;

  auto sys = make_DiscreteStateSpace(A, B, C, D, dt);

  /* State Space Simulation */
  for (std::size_t sim_step = 0; sim_step < 100; ++sim_step) {
    auto u = make_StateSpaceInput<1>(1.0);

    sys.update(u);

    std::cout << "X_0: " << sys.X(0, 0) << ", ";
    std::cout << "X_1: " << sys.X(1, 0) << ", ";
    std::cout << "Y: " << sys.Y(0, 0) << ", ";
    std::cout << std::endl;
  }
}
