#include <iostream>

#include "python_control.hpp"
#include "python_numpy.hpp"

#include <array>

using namespace PythonNumpy;
using namespace PythonControl;

constexpr std::size_t LKF_SIM_STEP_MAX = 50;

double get_lkf_test_input(std::size_t i, std::size_t j);

int main(void) {
  /* Create plant model */
  using SparseAvailable_A =
      SparseAvailable<ColumnAvailable<true, true, false, false>,
                      ColumnAvailable<false, true, true, false>,
                      ColumnAvailable<false, false, true, true>,
                      ColumnAvailable<false, false, false, true>>;

  auto A =
      make_SparseMatrix<SparseAvailable_A>(1.0, 0.1, 1.0, 0.1, 1.0, 0.1, 1.0);

  using SparseAvailable_B = SparseAvailable<
      ColumnAvailable<false, false>, ColumnAvailable<true, false>,
      ColumnAvailable<false, true>, ColumnAvailable<false, false>>;

  auto B = make_SparseMatrix<SparseAvailable_B>(0.1, 0.1);

  using SparseAvailable_C =
      SparseAvailable<ColumnAvailable<true, false, false, false>,
                      ColumnAvailable<false, false, true, false>>;

  auto C = make_SparseMatrix<SparseAvailable_C>(1.0, 1.0);

  auto D = make_SparseMatrixEmpty<double, 2, 2>();

  double dt = 0.1;

  constexpr std::size_t STATE_SIZE = decltype(A)::COLS;
  constexpr std::size_t INPUT_SIZE = decltype(B)::ROWS;
  constexpr std::size_t OUTPUT_SIZE = decltype(C)::COLS;

  auto sys = make_DiscreteStateSpace(A, B, C, D, dt);

  auto Q = make_DiagMatrix<STATE_SIZE>(1.0, 1.0, 1.0, 2.0);

  auto R = make_DiagMatrix<OUTPUT_SIZE>(10.0, 10.0);

  auto lkf = make_LinearKalmanFilter(sys, Q, R);

  /* prepare simulation */

  lkf.set_x_hat(make_StateSpaceState<STATE_SIZE>(0.0, 0.0, 0.0, 0.0));

  auto x_true = make_StateSpaceState<STATE_SIZE>(0.0, 0.0, 0.0, 0.1);
  auto y_measured = make_StateSpaceOutput<OUTPUT_SIZE>(0.0, 0.0);

  /* simulation */
  for (std::size_t i = 1; i < LKF_SIM_STEP_MAX; i++) {
    auto u = make_StateSpaceInput<INPUT_SIZE>(get_lkf_test_input(i - 1, 0),
                                              get_lkf_test_input(i - 1, 1));

    // system response
    x_true = A * x_true + B * u;
    y_measured = C * x_true + D * u;

    // kalman filter
    lkf.predict_and_update(u, y_measured);

    // results
    std::cout << "x_hat: ";
    std::cout << lkf.get_x_hat()(0, 0) << ",";
    std::cout << lkf.get_x_hat()(1, 0) << ",";
    std::cout << lkf.get_x_hat()(2, 0) << ",";
    std::cout << lkf.get_x_hat()(3, 0) << std::endl;
  }
  std::cout << std::endl;

  std::cout << "x_true: ";
  std::cout << x_true(0, 0) << ",";
  std::cout << x_true(1, 0) << ",";
  std::cout << x_true(2, 0) << ",";
  std::cout << x_true(3, 0) << std::endl;

  return 0;
}

Matrix<DefDense, double, LKF_SIM_STEP_MAX, 2> lkf_test_input(
    {{0.5, 0.5},   {0.5, 0.5},  {-0.5, -0.5}, {-0.5, 0.5}, {-0.5, -0.5},
     {0.5, 0.5},   {-0.5, 0.5}, {-0.5, 0.5},  {0.5, 0.5},  {0.5, -0.5},
     {-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5},  {0.5, -0.5}, {0.5, -0.5},
     {0.5, 0.5},   {0.5, 0.5},  {-0.5, -0.5}, {-0.5, 0.5}, {-0.5, -0.5},
     {0.5, 0.5},   {-0.5, 0.5}, {-0.5, 0.5},  {0.5, 0.5},  {0.5, -0.5},
     {-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5},  {0.5, -0.5}, {0.5, -0.5},
     {0.5, 0.5},   {0.5, 0.5},  {-0.5, -0.5}, {-0.5, 0.5}, {-0.5, -0.5},
     {0.5, 0.5},   {-0.5, 0.5}, {-0.5, 0.5},  {0.5, 0.5},  {0.5, -0.5},
     {-0.5, -0.5}, {0.5, -0.5}, {-0.5, 0.5},  {0.5, -0.5}, {0.5, -0.5},
     {0.5, 0.5},   {0.5, 0.5},  {-0.5, -0.5}, {-0.5, 0.5}, {-0.5, -0.5}});

double get_lkf_test_input(std::size_t i, std::size_t j) {

  return lkf_test_input(i, j);
}
