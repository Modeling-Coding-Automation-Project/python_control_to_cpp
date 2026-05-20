/**
 * @file discrete_state_space_lqi.cpp
 * @brief Example of discrete-time state-space LQI control simulation.
 *
 * This file demonstrates the use of a discrete-time state-space model and the
 * Linear Quadratic Integral (LQI) controller for a pendulum plant. The code
 * constructs sparse matrices for the plant's state-space representation, sets
 * up the LQI controller, and simulates the closed-loop system response over
 * time with reference tracking.
 */
#include <iostream>

#include "python_control.hpp"
#include "python_numpy.hpp"

using namespace PythonNumpy;
using namespace PythonControl;

constexpr std::size_t LQI_METHOD = PythonControl::LQR_METHOD_DARE;

/* Helper: set Eigen solver options (Arimoto-Potter only) */
template <
    std::size_t Method, typename LQI_T,
    typename std::enable_if<Method == LQR_METHOD_ARIMOTO_POTTER, int>::type = 0>
void set_lqi_solver_options(LQI_T &lqi, std::size_t q_ex_size) {
  lqi.set_Eigen_solver_iteration_max(q_ex_size);
  lqi.set_Eigen_solver_iteration_max_for_eigen_vector(3 * q_ex_size);
}

template <std::size_t Method, typename LQI_T,
          typename std::enable_if<Method == LQR_METHOD_DARE, int>::type = 0>
void set_lqi_solver_options(LQI_T &, std::size_t) {}

/* Dispatch: Arimoto-Potter uses continuous-time Ac, Bc, Cc */
template <
    std::size_t Method, typename AcType, typename BcType, typename CcType,
    typename AdType, typename BdType, typename QType, typename RType,
    typename std::enable_if<Method == LQR_METHOD_ARIMOTO_POTTER, int>::type = 0>
inline auto make_lqi_dispatch(const AcType &Ac, const BcType &Bc,
                              const CcType &Cc, const AdType &, const BdType &,
                              const QType &Q, const RType &R)
    -> decltype(make_LQI<Method>(Ac, Bc, Cc, Q, R)) {
  return make_LQI<Method>(Ac, Bc, Cc, Q, R);
}

/* Dispatch: DARE uses discrete-time Ad, Bd, Cc */
template <std::size_t Method, typename AcType, typename BcType, typename CcType,
          typename AdType, typename BdType, typename QType, typename RType,
          typename std::enable_if<Method == LQR_METHOD_DARE, int>::type = 0>
inline auto make_lqi_dispatch(const AcType &, const BcType &, const CcType &Cc,
                              const AdType &Ad, const BdType &Bd,
                              const QType &Q, const RType &R)
    -> decltype(make_LQI<Method>(Ad, Bd, Cc, Q, R)) {
  return make_LQI<Method>(Ad, Bd, Cc, Q, R);
}

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
  auto Cc = make_SparseMatrix<SparseAvailable_Cc>(1.0, 1.0);

  auto D = make_SparseMatrixEmpty<double, 2, 1>();

  double dt = 0.1;

  auto plant = make_DiscreteStateSpace(Ad, Bd, Cc, D, dt);

  /* Create controller model */
  constexpr std::size_t Q_EX_SIZE = 6;
  auto Q_ex = make_DiagMatrix<Q_EX_SIZE>(1.0, 0.1, 1.0, 0.1, 2.0, 0.1);
  auto R_ex = make_DiagMatrix<1>(1.0);

  auto lqi = make_lqi_dispatch<LQI_METHOD>(Ac, Bc, Cc, Ad, Bd, Q_ex, R_ex);

  set_lqi_solver_options<LQI_METHOD>(lqi, Q_EX_SIZE);

  auto K = lqi.solve();
  K = lqi.get_K();

  auto K_x =
      make_DenseMatrix<1, 4>(K.template get<0, 0>(), K.template get<0, 1>(),
                             K.template get<0, 2>(), K.template get<0, 3>());
  auto K_e =
      make_DenseMatrix<1, 2>(K.template get<0, 4>(), K.template get<0, 5>());

  std::cout << "K: ";
  std::cout << K(0, 0) << ", ";
  std::cout << K(0, 1) << ", ";
  std::cout << K(0, 2) << ", ";
  std::cout << K(0, 3) << ", ";
  std::cout << K(0, 4) << ", ";
  std::cout << K(0, 5) << ", ";
  std::cout << std::endl << std::endl;

  /* Simulation */
  auto X_ref = make_DenseMatrix<4, 1>(1.0, 0.0, 0.0, 0.0);
  auto Y_ref = make_DenseMatrix<2, 1>(1.0, 0.0);
  auto e_y_integral = make_DenseMatrix<2, 1>(0.0, 0.0);

  for (int i = 0; i < 100; i++) {
    auto X = plant.get_X();
    auto Y = Cc * X;
    auto e_y = Y_ref - Y;
    e_y_integral = e_y_integral + e_y * dt;

    auto U = K_x * (X_ref - X) + K_e * e_y_integral;

    plant.update(U);

    std::cout << "X_0: " << X(0, 0) << ", ";
    std::cout << "X_2: " << X(2, 0) << ", ";
    std::cout << std::endl;
  }

  return 0;
}
