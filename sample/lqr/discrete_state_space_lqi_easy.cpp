#include <iostream>

#include "python_control.hpp"
#include "python_numpy.hpp"

using namespace PythonNumpy;
using namespace PythonControl;

int main(void) {
  /* Create plant model */
  using SparseAvailable_Ac = SparseAvailable<ColumnAvailable<true, true>,
                                             ColumnAvailable<true, false>>;

  auto Ac = make_SparseMatrix<SparseAvailable_Ac>(-2.0, -1.0, 1.0);

  using SparseAvailable_Bc =
      SparseAvailable<ColumnAvailable<true>, ColumnAvailable<false>>;

  auto Bc = make_SparseMatrix<SparseAvailable_Bc>(1.0);

  using SparseAvailable_Cc = SparseAvailable<ColumnAvailable<false, true>>;

  auto Cc = make_SparseMatrix<SparseAvailable_Cc>(1.0);

  double dt = 0.1;

  auto Dc = make_SparseMatrixEmpty<double, 1, 1>();

  constexpr std::size_t State_Num = decltype(Ac)::COLS;
  constexpr std::size_t Input_Num = decltype(Bc)::ROWS;
  constexpr std::size_t Output_Num = decltype(Cc)::COLS;

  /* Discretize plant model */
  auto Ad = make_DiagMatrixIdentity<double, State_Num>() + Ac * dt;
  auto Bd = Bc * dt;
  auto Cd = Cc;
  auto Dd = Dc;

  auto plant = make_DiscreteStateSpace(Ad, Bd, Cd, Dd, dt);

  /* Create controller model */
  constexpr std::size_t Q_EX_SIZE = State_Num + Output_Num;
  auto Q_ex = make_DiagMatrix<Q_EX_SIZE>(0.0, 2.0, 2.0);

  auto R_ex = make_DiagMatrix<Input_Num>(1.0);

  auto lqi = make_LQI(Ac, Bc, Cc, Q_ex, R_ex);

  lqi.set_Eigen_solver_iteration_max(Q_EX_SIZE);
  lqi.set_Eigen_solver_iteration_max_for_eigen_vector(3 * Q_EX_SIZE);

  auto K = lqi.solve();
  K = lqi.get_K();

  auto K_x =
      make_DenseMatrix<1, 2>(K.template get<0, 0>(), K.template get<0, 1>());
  auto K_e = make_DenseMatrix<1, 1>(K.template get<0, 2>());

  std::cout << "K: ";
  std::cout << K(0, 0) << ", ";
  std::cout << K(0, 1) << ", ";
  std::cout << K(0, 2);
  std::cout << std::endl << std::endl;

  /* Simulation */
  auto x_ref = make_StateSpaceState<State_Num>(0.0, 1.0);
  auto y_ref = make_StateSpaceOutput<Output_Num>(1.0);

  auto e_y_integral = make_StateSpaceOutput<Output_Num>(0.0);

  for (int i = 0; i < 100; i++) {
    auto x = plant.get_X();
    auto y = Cc * x;
    auto e_y = y_ref - y;
    e_y_integral = e_y_integral + e_y * dt;

    auto u = K_x * (x_ref - x) + K_e * e_y_integral;

    plant.update(u);

    std::cout << "x(1, 0): " << x(1, 0);
    std::cout << std::endl;
  }

  return 0;
}
