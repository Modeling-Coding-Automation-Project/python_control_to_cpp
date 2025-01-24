#include <iostream>

#include "python_control.hpp"
#include "python_numpy.hpp"

using namespace PythonNumpy;
using namespace PythonControl;

int main(void) {
  /* Create plant model */
  using SparseAvailable_Ac =
      SparseAvailable<ColumnAvailable<false, true, false, false>,
                      ColumnAvailable<false, true, true, false>,
                      ColumnAvailable<false, false, false, true>,
                      ColumnAvailable<false, true, true, false>>;
  auto Ac =
      make_SparseMatrix<SparseAvailable_Ac>(1.0, -0.1, 3.0, 1.0, -0.5, 30.0);

  using SparseAvailable_Ad =
      SparseAvailable<ColumnAvailable<true, true, false, false>,
                      ColumnAvailable<false, true, true, false>,
                      ColumnAvailable<false, false, true, true>,
                      ColumnAvailable<false, true, true, true>>;
  auto Ad = make_SparseMatrix<SparseAvailable_Ad>(1.0, 0.1, 0.99, 0.3, 1.0, 0.1,
                                                  -0.05, 3.0, 1.0);

  using SparseAvailable_Bc =
      SparseAvailable<ColumnAvailable<false>, ColumnAvailable<true>,
                      ColumnAvailable<false>, ColumnAvailable<true>>;
  auto Bc = make_SparseMatrix<SparseAvailable_Bc>(2.0, 5.0);

  using SparseAvailable_Bd =
      SparseAvailable<ColumnAvailable<false>, ColumnAvailable<true>,
                      ColumnAvailable<false>, ColumnAvailable<true>>;
  auto Bd = make_SparseMatrix<SparseAvailable_Bd>(0.2, 0.5);

  using SparseAvailable_Cc =
      SparseAvailable<ColumnAvailable<true, false, false, false>,
                      ColumnAvailable<false, false, true, false>>;
  auto C = make_SparseMatrix<SparseAvailable_Cc>(1.0, 1.0);

  auto D = make_SparseMatrixEmpty<double, 2, 1>();

  double dt = 0.1;

  auto plant = make_DiscreteStateSpace(Ad, Bd, C, D, dt);

  /* Create controller model */
  auto Q = make_DiagMatrix<4>(1.0, 0.0, 1.0, 0.0);
  auto R = make_DiagMatrix<1>(1.0);

  auto lqr = make_LQR(Ac, Bc, Q, R);

  auto K = lqr.solve();
  K = lqr.get_K();

  /* Simulation */
  auto X_ref = make_DenseMatrix<4, 1>(1.0, 0.0, 0.0, 0.0);

  for (int i = 0; i < 100; i++) {
    auto X = plant.get_X();
    auto U = K * (X_ref - X);

    plant.update(U);

    std::cout << "X_0: " << X(0, 0) << ", ";
    std::cout << "X_2: " << X(2, 0) << ", ";
    std::cout << std::endl;
  }

  return 0;
}
